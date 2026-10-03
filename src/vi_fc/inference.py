"""FunctionCaller: câu nói + trạng thái xe -> Action (gọi tool / hỏi lại / từ chối / trả lời).

Backend:
- "llama_cpp": GGUF chạy CPU qua llama-cpp-python (mặc định, dùng trong viet-copilot)
- "openai":    server bất kỳ kiểu OpenAI (llama.cpp server, Groq...) - dự phòng khi máy yếu
- "hf":        transformers trên GPU, chỉ để eval trên Kaggle (base / SFT / DPO trước khi export)
- "gemini":    Gemini API function calling, chỉ để eval làm cận trên
Các thư viện nặng đều import bên trong hàm: CI và laptop không cài vẫn import được module này.

Sau khi model trả lời, guard kiểm tra lại (phòng thủ nhiều lớp): tool không tồn tại / sai schema thì
không thực thi, vi phạm luật an toàn thì ghi đè thành từ chối dù model đã gọi tool.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass

from .parse import Action, from_openai_message, parse_output
from .prompt import SYSTEM_PROMPT, build_messages, render_chat_template
from .safety import check_calls
from .state import as_state
from .tools import openai_tools, validate_call

log = logging.getLogger(__name__)

UNCLEAR_TEXT = "Mình chưa rõ yêu cầu, bạn nói lại giúp mình được không?"


@dataclass
class GenResult:
    text: str
    message: dict | None = None  # server đã tách sẵn tool_calls (API kiểu OpenAI / Gemini)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LlamaCppBackend:
    def __init__(self, model_path: str | None = None, n_threads: int | None = None, n_ctx: int = 4096,
                 chat_template: str | None = None, llm=None, verbose: bool = False):
        if llm is None:
            from llama_cpp import Llama

            llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=verbose)
        self.llm = llm
        # tự render bằng template nhúng trong GGUF thay vì create_chat_completion: chat handler của
        # llama-cpp-python không truyền enable_thinking=False, Qwen3 sẽ "nghĩ" và chậm gấp mấy lần
        self.template = chat_template or getattr(llm, "metadata", {}).get("tokenizer.chat_template")
        if not self.template:
            raise ValueError("GGUF không có tokenizer.chat_template, truyền chat_template= vào")

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int, temperature: float) -> GenResult:
        prompt = render_chat_template(self.template, messages, tools, add_generation_prompt=True,
                                      enable_thinking=False)
        # prefix (system + tools) giống nhau giữa các lượt -> llama.cpp tự dùng lại KV cache, chỉ prefill phần mới
        out = self.llm.create_completion(prompt, max_tokens=max_tokens, temperature=temperature,
                                         stop=["<|im_end|>", "<|endoftext|>"])
        usage = out.get("usage", {})
        return GenResult(out["choices"][0]["text"], None, usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0))


def _to_api_messages(messages: list[dict]) -> list[dict]:
    """Message nội bộ -> chuẩn OpenAI: arguments phải là chuỗi JSON, tool_call cần id."""
    out = []
    for i, m in enumerate(messages):
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {"id": f"call_{i}_{j}", "type": "function",
                 "function": {"name": tc["function"]["name"],
                              "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                for j, tc in enumerate(m["tool_calls"])
            ]
        out.append(m)
    return out


class OpenAIBackend:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, extra_body: dict | None = None,
                 timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # llama.cpp server: {"chat_template_kwargs": {"enable_thinking": False}}; Groq báo lỗi nếu gửi field lạ
        self.extra_body = extra_body or {}
        self.timeout = timeout

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int, temperature: float) -> GenResult:
        import requests

        body = {"model": self.model, "messages": _to_api_messages(messages), "tools": tools,
                "temperature": temperature, "max_tokens": max_tokens, **self.extra_body}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return GenResult(msg.get("content") or "", msg, usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0))


class HFBackend:
    """transformers + (tuỳ chọn) LoRA adapter. Truyền sẵn model/tokenizer đã load (vd từ Unsloth) cũng được."""

    def __init__(self, model, tokenizer=None, adapter: str | None = None, load_in_4bit: bool = False):
        if isinstance(model, str):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
            kw = {"torch_dtype": torch.float16, "device_map": "auto"}
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(model, **kw)
            if adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter)
            model.eval()
        self.model, self.tokenizer = model, tokenizer

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int, temperature: float) -> GenResult:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                    enable_thinking=False, tokenize=False)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        kw = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kw["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**enc, **kw, pad_token_id=self.tokenizer.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        # giữ special token: cần thấy <tool_call>; <|im_end|> sẽ bị parse bỏ đi
        text = self.tokenizer.decode(new, skip_special_tokens=False)
        return GenResult(text, None, int(enc["input_ids"].shape[1]), int(new.shape[0]))


_GEMINI_SCHEMA_KEYS = {"type", "description", "enum", "properties", "required", "minimum", "maximum", "items"}


def _gemini_schema(s: dict) -> dict:
    out = {}
    for k, v in s.items():
        if k not in _GEMINI_SCHEMA_KEYS:
            continue
        if k == "type":
            v = v.upper()
        elif k == "properties":
            v = {pk: _gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            v = _gemini_schema(v)
        out[k] = v
    return out


def gemini_tools(tools: list[dict]) -> list[dict]:
    decls = []
    for t in tools:
        f = t["function"]
        d = {"name": f["name"], "description": f["description"]}
        if f["parameters"].get("properties"):  # Gemini không nhận OBJECT rỗng
            d["parameters"] = _gemini_schema(f["parameters"])
        decls.append(d)
    return [{"functionDeclarations": decls}]


def gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    system, contents = "", []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        else:
            parts = [{"text": m["content"]}] if m.get("content") else []
            parts += [{"functionCall": {"name": tc["function"]["name"], "args": tc["function"]["arguments"]}}
                      for tc in m.get("tool_calls", [])]
            contents.append({"role": "model", "parts": parts})
    return system, contents


def gemini_message(data: dict) -> dict:
    """Response Gemini -> message dạng OpenAI để dùng chung from_openai_message."""
    try:
        parts = data["candidates"][0]["content"].get("parts", [])
    except (KeyError, IndexError):
        return {"content": ""}
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    calls = [{"type": "function", "function": {"name": p["functionCall"]["name"],
                                               "arguments": p["functionCall"].get("args", {})}}
             for p in parts if "functionCall" in p]
    return {"content": text, "tool_calls": calls}


class GeminiBackend:
    """Chỉ dùng để đánh giá: điều khoản Gemini API không cho dùng output để train model khác."""

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: int = 60):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.timeout = timeout

    def generate(self, messages: list[dict], tools: list[dict], max_tokens: int, temperature: float) -> GenResult:
        import requests

        system, contents = gemini_contents(messages)
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                "tools": gemini_tools(tools), "generationConfig": {"temperature": temperature}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        r = requests.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usageMetadata", {})
        msg = gemini_message(data)
        return GenResult(msg["content"], msg, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0))


def guard_action(action: Action, state, confirmed: bool = False) -> Action:
    """Lớp chặn sau model. Trả về Action mới, Action gốc của model lưu ở meta['model_action']."""
    if action.kind != "call":
        return action
    state = as_state(state)
    notes = list(action.notes)
    valid = []
    for c in action.calls:
        errs = validate_call(c.name, c.arguments)
        if errs:
            notes.append("guard bỏ lệnh sai schema: " + "; ".join(errs))
        else:
            valid.append(c)
    meta = {**action.meta, "model_action": action.to_dict()}
    if not valid:
        # không có lệnh nào chạy được (tool bịa, thiếu tham số...) -> hỏi lại thay vì làm bừa
        return Action("ask", [], UNCLEAR_TEXT, action.valid_format, notes, meta, action.raw)
    verdict = check_calls(valid, state)
    if verdict.decision == "refuse" or (verdict.decision == "confirm" and not confirmed):
        kind = "refuse" if verdict.decision == "refuse" else "ask"
        log.warning("guard ghi đè %s -> %s (%s)", [c.name for c in valid], kind, verdict.rule)
        return Action(kind, [], verdict.message, action.valid_format, notes + [f"guard: {verdict.rule}"], meta,
                      action.raw)
    if len(valid) == len(action.calls):
        return action
    return Action("call", valid, action.text, action.valid_format, notes, meta, action.raw)


class FunctionCaller:
    """Dùng trong viet-copilot:

        fc = FunctionCaller("llama_cpp", "models/vi-fc-qwen3-1.7b-Q4_K_M.gguf", n_threads=6)
        action = fc.decide("bật điều hoà 22 độ", {"speed_kmh": 40})
        if action.kind == "call":
            for name, args in action.calls: ...
    """

    def __init__(self, backend: str = "llama_cpp", model=None, *, n_threads: int | None = None, n_ctx: int = 4096,
                 base_url: str | None = None, api_key: str | None = None, extra_body: dict | None = None,
                 tokenizer=None, adapter: str | None = None, guard: bool = True, max_tokens: int = 256,
                 temperature: float = 0.0, system_prompt: str = SYSTEM_PROMPT, tool_names: list[str] | None = None):
        self.backend_name = backend if isinstance(backend, str) else type(backend).__name__
        if not isinstance(backend, str):
            self.backend = backend  # object có .generate(), dùng cho test
        elif backend == "llama_cpp":
            self.backend = LlamaCppBackend(model or os.environ["VI_FC_GGUF"], n_threads=n_threads, n_ctx=n_ctx)
        elif backend == "openai":
            self.backend = OpenAIBackend(base_url or "http://127.0.0.1:8080/v1", model or "local", api_key,
                                         extra_body)
        elif backend == "hf":
            self.backend = HFBackend(model, tokenizer, adapter)
        elif backend == "gemini":
            self.backend = GeminiBackend(model, api_key)
        else:
            raise ValueError(f"backend không hỗ trợ: {backend}")
        self.guard = guard
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.tools = openai_tools(tool_names)

    def decide(self, user_text: str, state=None, history: list[dict] | None = None,
               confirmed: bool = False) -> Action:
        """state: VehicleState hoặc dict. history: các lượt trước [{"role", "content"}].
        confirmed=True khi người dùng vừa xác nhận câu hỏi lại của guard (vd mở cửa sổ khi chạy nhanh)."""
        messages = build_messages(user_text, state, history, self.system_prompt)
        t0 = time.perf_counter()
        gen = self.backend.generate(messages, self.tools, self.max_tokens, self.temperature)
        dt = time.perf_counter() - t0
        action = from_openai_message(gen.message) if gen.message is not None else parse_output(gen.text)
        action.meta.update({
            "backend": self.backend_name, "latency_s": round(dt, 3), "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            # tính cả thời gian prefill nên thấp hơn tốc độ decode thuần
            "tokens_per_s": round(gen.completion_tokens / dt, 1) if dt > 0 else 0.0,
        })
        if self.guard:
            action = guard_action(action, state, confirmed)
        return action

    def warmup(self) -> float:
        """Prefill sẵn system + tools (~1.5k token) lúc khởi động để câu đầu tiên của người dùng không bị chậm."""
        t0 = time.perf_counter()
        self.backend.generate(build_messages("xin chào", None, None, self.system_prompt), self.tools, 1, 0.0)
        return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="thử nhanh: python -m vi_fc.inference 'bật điều hoà 22 độ' --speed 40")
    ap.add_argument("text")
    ap.add_argument("--backend", default="llama_cpp")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--speed", type=float, default=0)
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--no-guard", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    fc = FunctionCaller(args.backend, args.model, n_threads=args.threads, base_url=args.base_url,
                        guard=not args.no_guard)
    print(f"warmup {fc.warmup():.1f}s")
    a = fc.decide(args.text, {"speed_kmh": args.speed, "gear": "D" if args.speed else "P", "is_night": args.night})
    print(json.dumps({**a.to_dict(), "meta": a.meta}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
