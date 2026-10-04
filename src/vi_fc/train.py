"""Phần dùng chung cho notebook 02 (SFT) và 03 (DPO).

Unsloth nhanh hơn ~2 lần, tốn ít VRAM hơn nhưng hay vỡ mỗi khi Colab đổi bản torch / Python, nên hàm nào cũng có
nhánh transformers + peft + bitsandbytes thuần. Thư viện nặng import trong hàm như inference.py.
"""

import gc
import math
import os
import subprocess
import sys
import time

BASE = "Qwen/Qwen3-1.7B"
UNSLOTH_BASE = "unsloth/Qwen3-1.7B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# token/giây (quy về SFT) trên T4 với chuỗi ~1.7k token, chỉ để chia ngân sách thời gian; lệch thì time_budget chặn
TPS = {True: 1500, False: 700}


def has_unsloth() -> bool:
    """Thử import ở process riêng: import hỏng giữa chừng trong kernel để lại transformers bị patch dở."""
    if os.environ.get("VI_FC_NO_UNSLOTH"):
        return False
    r = subprocess.run([sys.executable, "-c", "import unsloth"], capture_output=True, text=True)
    if r.returncode:
        print("không dùng được unsloth -> transformers + peft:", (r.stderr.strip().splitlines() or ["?"])[-1])
    return r.returncode == 0


def fp_flags() -> dict:
    """T4 (7.5) không có bf16 thật, nhưng torch.cuda.is_bf16_supported() vẫn trả True vì tính cả giả lập."""
    import torch

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    return {"bf16": bf16, "fp16": not bf16}


def load_4bit(name: str, max_len: int, unsloth: bool):
    import torch

    if unsloth:
        from unsloth import FastLanguageModel

        return FastLanguageModel.from_pretrained(name, max_seq_length=max_len, load_in_4bit=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .inference import dtype_kw

    dt = torch.bfloat16 if fp_flags()["bf16"] else torch.float16
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=dt)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0}, **dtype_kw(dt))
    return model, AutoTokenizer.from_pretrained(name)


def add_lora(model, unsloth: bool, r: int = 16, alpha: int = 32, seed: int = 0):
    if unsloth:
        from unsloth import FastLanguageModel

        model = FastLanguageModel.get_peft_model(
            model, r=r, lora_alpha=alpha, lora_dropout=0, bias="none", target_modules=LORA_TARGETS,
            use_gradient_checkpointing="unsloth", random_state=seed,
        )
        FastLanguageModel.for_training(model)
        return model
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(seed)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    return get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))


def set_mode(model, unsloth: bool, train: bool):
    if unsloth:
        from unsloth import FastLanguageModel

        (FastLanguageModel.for_training if train else FastLanguageModel.for_inference)(model)
        return
    model.train(train)
    model.config.use_cache = not train


def tokenize_sft(tokenizer, prompt: str, completion: str, max_len: int) -> dict | None:
    """Loss chỉ tính trên completion (lượt trả lời cuối). Dài quá max_len thì bỏ: cắt là mất chính câu trả lời."""
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        return None
    return {"input_ids": p + c, "attention_mask": [1] * (len(p) + len(c)), "labels": [-100] * len(p) + c}


def plan_steps(n: int, per_step: int, epochs: float, tokens_per_item: float, tps: float, minutes: float) -> int:
    """Số step = min(đủ số epoch, vừa ngân sách phút). Cố định từ đầu để lịch LR cosine chạy hết."""
    full = math.ceil(n * epochs / per_step)
    fit = int(minutes * 60 * tps / (per_step * tokens_per_item))
    return max(1, min(full, fit))


def time_budget(minutes: float):
    """Callback chặn cứng: ước lượng tốc độ sai thì vẫn dừng đúng giờ (có lưu checkpoint nếu đang bật lưu)."""
    from transformers import TrainerCallback

    class TimeBudget(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            if time.time() - self.t0 > minutes * 60:
                print(f"hết {minutes} phút ở step {state.global_step}/{state.max_steps}, dừng")
                control.should_training_stop = True
                control.should_save = args.save_strategy != "no"
            return control

    return TimeBudget()


def merge_lora(adapter_dir: str, out_dir: str, base: str = BASE):
    """Gộp LoRA vào trọng số fp16 bằng peft, không cần Unsloth. Adapter train trên bản 4-bit vẫn gộp vào bản
    16-bit của cùng model (cách làm chuẩn của QLoRA)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .inference import dtype_kw

    model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0} if torch.cuda.is_available() else None,
                                                 **dtype_kw(torch.float16))
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
