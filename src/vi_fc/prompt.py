"""Dựng messages cho model + render chat template.

Định dạng giữ đúng chat template gốc của Qwen3 (tools= ..., enable_thinking=False) để model không phải
học lại cú pháp gọi hàm, chỉ học "gọi cái gì, khi nào".
"""

import json

from .parse import Action
from .state import VehicleState, as_state

SYSTEM_PROMPT = """Bạn là trợ lý giọng nói trên ô tô điện. Mỗi lượt người dùng có kèm trạng thái xe trong thẻ <xe>.
- Đủ thông tin: gọi một hoặc nhiều hàm phù hợp, không nói thêm.
- Thiếu thông tin bắt buộc: hỏi lại đúng một câu ngắn.
- Yêu cầu không an toàn với trạng thái xe hiện tại: từ chối ngắn gọn và nêu lý do.
- Hỏi về luật giao thông hoặc biển báo: gọi lookup_traffic_law hoặc lookup_sign.
- Ngoài phạm vi: trả lời ngắn, thân thiện.
Chỉ dùng các hàm được cung cấp, đúng tên và đúng kiểu tham số."""


def user_turn(text: str, state: VehicleState | dict | None) -> str:
    return f"<xe>{as_state(state).render()}</xe>\n{text.strip()}"


def build_messages(user_text: str, state=None, history: list[dict] | None = None,
                   system: str = SYSTEM_PROMPT) -> list[dict]:
    """history: các lượt trước dạng {"role": "user"|"assistant", "content": ...} (user thô, chưa có <xe>).
    Chỉ lượt cuối mới gắn state: state cũ không còn đúng, để vào chỉ tốn token và gây nhiễu."""
    msgs = [{"role": "system", "content": system}]
    msgs += [dict(m) for m in history or []]
    msgs.append({"role": "user", "content": user_turn(user_text, state)})
    return msgs


def assistant_message(action: Action) -> dict:
    msg: dict = {"role": "assistant", "content": action.text}
    if action.kind == "call" and action.calls:
        msg["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                             for c in action.calls]
    return msg


def record_messages(rec: dict, with_gold: bool = True) -> list[dict]:
    """Bản ghi dữ liệu (xem datagen) -> messages. with_gold=True thì thêm lượt assistant đúng ở cuối (để SFT)."""
    msgs = build_messages(rec["user"], rec.get("state"), rec.get("history"))
    if with_gold:
        msgs.append(assistant_message(Action.from_dict(rec["gold"])))
    return msgs


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # tojson mặc định của jinja2 escape HTML và ensure_ascii -> tiếng Việt thành ệ..., khác hẳn lúc train
    # (transformers override filter này y như dưới). Lệch chỗ này là model chạy GGUF kém hơn lúc eval trên GPU.
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def render_chat_template(template: str, messages: list[dict], tools: list[dict] | None = None,
                         add_generation_prompt: bool = True, enable_thinking: bool = False, **extra) -> str:
    """Render chat template Jinja giống transformers.apply_chat_template, dùng cho llama.cpp
    (lấy template nhúng trong GGUF) để không cần cài transformers trên laptop."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking, **extra,
    )


def split_prompt_completion(apply_template, messages: list[dict]) -> tuple[str, str]:
    """Tách (prompt, completion) cho SFT/DPO. `apply_template(messages, add_generation_prompt) -> str`
    (bọc tokenizer.apply_chat_template, đã gắn sẵn tools). Lấy full trừ prompt thay vì tự ghép chuỗi
    completion, để chắc chắn khớp từng ký tự với template (Qwen3 chèn <think>\\n\\n</think> trước câu trả lời cuối)."""
    prompt = apply_template(messages[:-1], True)
    full = apply_template(messages, False)
    if not full.startswith(prompt):
        raise ValueError("chat template: full không bắt đầu bằng prompt, kiểm tra enable_thinking")
    return prompt, full[len(prompt):]
