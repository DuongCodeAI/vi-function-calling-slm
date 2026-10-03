"""Đọc output của model thành `Action`.

Qwen3 sinh `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (có thể nhiều cái liền nhau).
Ngoài ra chấp nhận JSON trần (model base / model khác hay quên thẻ) và câu bị cắt cụt do max_tokens.
Không có tool call thì là câu chữ: phân loại thành hỏi lại (ask), từ chối (refuse) hay trả lời thường (reply).

Giới hạn của phân loại bằng từ khoá (ghi rõ vì nó ảnh hưởng tới số eval):
- câu từ chối viết dạng hỏi ("Bạn dừng xe rồi mình mở nhé?") sẽ thành ask;
- câu ngoài phạm vi có "xin lỗi, mình không thể..." sẽ thành refuse.
Dữ liệu train được kiểm tra để text gold luôn phân loại đúng (xem datagen/filters.py), nên model đã
fine-tune học đúng "quy ước" này; model base và Gemini thì không -> số kind-accuracy của chúng bị thiệt một chút.
"""

import json
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from .text import normalize

KINDS = ("call", "ask", "refuse", "reply")


class ToolCall(NamedTuple):
    name: str
    arguments: dict


@dataclass
class Action:
    kind: str  # call | ask | refuse | reply
    calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    valid_format: bool = True
    notes: list[str] = field(default_factory=list)  # cảnh báo parse, guard ghi đè...
    meta: dict = field(default_factory=dict)  # latency, số token...
    raw: str = ""

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "calls": [{"name": c.name, "arguments": c.arguments} for c in self.calls],
             "text": self.text}
        if not self.valid_format:
            d["valid_format"] = False
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        calls = [ToolCall(c["name"], c.get("arguments") or {}) for c in d.get("calls", [])]
        return cls(d["kind"], calls, d.get("text", ""), d.get("valid_format", True), list(d.get("notes", [])))


_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN = re.compile(r"<tool_call>(.*)$", re.DOTALL)  # bị cắt do hết max_tokens
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _load_json(s: str):
    s = _FENCE.sub("", s.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # rác phía sau JSON (vd "}}\n" thừa hoặc lời giải thích) -> lấy object đầu tiên
    i = min((p for p in (s.find("{"), s.find("[")) if p >= 0), default=-1)
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[i:])[0]
    except json.JSONDecodeError:
        return None


def _to_call(obj) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # dạng OpenAI {"type": "function", "function": {...}}
        obj = obj["function"]
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):  # API kiểu OpenAI trả arguments là chuỗi JSON
        args = _load_json(args) if args.strip() else {}
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        return None
    return ToolCall(name, args)


def _calls_from(obj) -> list[ToolCall] | None:
    items = obj if isinstance(obj, list) else [obj]
    calls = [_to_call(o) for o in items]
    if not calls or any(c is None for c in calls):
        return None
    return calls


_STRONG_REFUSE = ("không an toàn", "nguy hiểm", "không được phép", "không cho phép")
_WEAK_REFUSE = ("không thể", "xin lỗi", "từ chối", "không làm được", "chưa làm được", "không hỗ trợ")
_QUESTION_END = re.compile(
    r"(nào|bao nhiêu|mấy|đâu|gì|ai|bao lâu|không|chưa|chứ)\s*(ạ|vậy|nhỉ|thế|nhé)?\s*[.!…]*$"
)


def classify_text(text: str) -> str:
    """ask / refuse / reply cho câu trả lời không gọi tool. Thứ tự ưu tiên có chủ ý:
    từ chối vì an toàn > câu hỏi > từ chối chung chung > trả lời thường."""
    t = normalize(text, keep_diacritics=True)
    if not t:
        return "reply"
    if any(k in t for k in _STRONG_REFUSE):
        return "refuse"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip().lower()) if s.strip()]
    if any(s.endswith("?") for s in sentences) or (sentences and _QUESTION_END.search(sentences[-1])):
        return "ask"
    if any(k in t for k in _WEAK_REFUSE):
        return "refuse"
    return "reply"


def parse_output(raw: str) -> Action:
    text = _THINK.sub("", raw or "").replace("<think>", "").replace("</think>", "").strip()
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    blocks = _TOOL_CALL.findall(text)
    rest = _TOOL_CALL.sub("", text)
    if (m := _TOOL_CALL_OPEN.search(rest)) is not None:
        blocks.append(m.group(1))
        rest = rest[: m.start()]
    rest = rest.strip()

    if blocks:
        calls, bad = [], 0
        for b in blocks:
            parsed = _calls_from(_load_json(b))
            if parsed is None:
                bad += 1
            else:
                calls += parsed
        notes = [f"{bad} tool_call không parse được"] if bad else []
        return Action("call", calls, rest, valid_format=bad == 0, notes=notes, raw=raw)

    # không có thẻ: thử JSON trần
    if rest.startswith(("{", "[", "```")):
        obj = _load_json(rest)
        calls = _calls_from(obj) if obj is not None else None
        if calls:
            return Action("call", calls, "", notes=["json trần, không có <tool_call>"], raw=raw)
        return Action(classify_text(rest), [], rest, valid_format=False, notes=["json không đúng dạng tool call"],
                      raw=raw)

    return Action(classify_text(rest), [], rest, raw=raw)


def from_openai_message(msg: dict) -> Action:
    """Message trả về từ /chat/completions (llama.cpp server, Groq...). Server đã tách tool_calls thì dùng
    luôn, không thì parse content (llama.cpp không bật --jinja sẽ để nguyên <tool_call> trong content)."""
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return parse_output(content)
    calls = [_to_call(tc) for tc in tcs]
    ok = [c for c in calls if c is not None]
    bad = len(calls) - len(ok)
    return Action("call", ok, content.strip(), valid_format=bad == 0,
                  notes=[f"{bad} tool_call không parse được"] if bad else [], raw=json.dumps(msg, ensure_ascii=False))
