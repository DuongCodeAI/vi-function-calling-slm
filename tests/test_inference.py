import json

import pytest
from test_parse import MINI_TEMPLATE

from vi_fc.inference import (
    UNCLEAR_TEXT,
    FunctionCaller,
    GenResult,
    LlamaCppBackend,
    _to_api_messages,
    gemini_contents,
    gemini_message,
    gemini_tools,
    guard_action,
)
from vi_fc.parse import Action, ToolCall
from vi_fc.tools import openai_tools

MOVING = {"speed_kmh": 50, "gear": "D", "is_night": True}


def test_guard_overrides_unsafe_call():
    a = Action("call", [ToolCall("unlock_doors", {})])
    g = guard_action(a, MOVING)
    assert g.kind == "refuse" and "không an toàn" in g.text
    assert g.meta["model_action"]["calls"][0]["name"] == "unlock_doors"
    assert guard_action(a, {"speed_kmh": 0}) is a


def test_guard_confirm():
    a = Action("call", [ToolCall("open_window", {"position": "all"})])
    assert guard_action(a, {"speed_kmh": 110}).kind == "ask"
    assert guard_action(a, {"speed_kmh": 110}, confirmed=True).kind == "call"


def test_guard_schema():
    a = Action("call", [ToolCall("turn_on_ac", {})])
    g = guard_action(a, {})
    assert g.kind == "ask" and g.text == UNCLEAR_TEXT and "tool không tồn tại" in g.notes[0]
    multi = Action("call", [ToolCall("set_volume", {"level": 99}), ToolCall("lock_doors", {})])
    g = guard_action(multi, {})
    assert g.kind == "call" and g.calls == [ToolCall("lock_doors", {})]
    assert guard_action(Action("reply", text="chào"), MOVING).kind == "reply"


class FakeBackend:
    def __init__(self, text):
        self.text = text
        self.seen = None

    def generate(self, messages, tools, max_tokens, temperature):
        self.seen = messages
        return GenResult(self.text, None, 1500, 20)


def test_function_caller_with_fake_backend():
    fb = FakeBackend('<tool_call>\n{"name": "unlock_doors", "arguments": {}}\n</tool_call>')
    fc = FunctionCaller(fb)
    a = fc.decide("mở khoá cửa", MOVING)
    assert a.kind == "refuse"
    assert fb.seen[-1]["content"].startswith("<xe>tốc độ 50 km/h")
    assert a.meta["prompt_tokens"] == 1500 and "latency_s" in a.meta
    raw = FunctionCaller(fb, guard=False).decide("mở khoá cửa", MOVING)
    assert raw.kind == "call"


def test_unknown_backend():
    with pytest.raises(ValueError):
        FunctionCaller("ollama")


class FakeLlama:
    metadata = {"tokenizer.chat_template": MINI_TEMPLATE}

    def __init__(self):
        self.prompt = None

    def create_completion(self, prompt, **kw):
        self.prompt = prompt
        return {"choices": [{"text": '<tool_call>\n{"name": "set_volume", "arguments": {"level": 7}}\n</tool_call>'}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 18}}


def test_llama_cpp_backend_renders_template():
    llm = FakeLlama()
    fc = FunctionCaller(LlamaCppBackend(llm=llm))
    a = fc.decide("vặn loa mức 7", {})
    assert a.calls == [ToolCall("set_volume", {"level": 7})]
    assert llm.prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Chỉnh âm lượng loa." in llm.prompt  # tools nằm trong prompt, không bị escape \u


def test_to_api_messages():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "set_volume", "arguments": {"level": 3}}}]}]
    out = _to_api_messages(msgs)
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"level": 3}
    assert out[0]["tool_calls"][0]["id"]
    assert isinstance(msgs[0]["tool_calls"][0]["function"]["arguments"], dict)  # không sửa input


def test_gemini_conversion():
    decls = gemini_tools(openai_tools(["set_lights", "lock_doors", "navigate_to"]))[0]["functionDeclarations"]
    assert decls[0]["parameters"]["properties"]["mode"]["type"] == "STRING"
    assert "parameters" not in decls[1]
    assert "minLength" not in decls[2]["parameters"]["properties"]["destination"]
    system, contents = gemini_contents([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    assert system == "sys" and contents[0]["role"] == "user"
    data = {"candidates": [{"content": {"parts": [
        {"text": "nghĩ...", "thought": True},
        {"functionCall": {"name": "set_lights", "args": {"mode": "low"}}}]}}]}
    msg = gemini_message(data)
    assert msg["content"] == "" and msg["tool_calls"][0]["function"]["arguments"] == {"mode": "low"}
    assert gemini_message({"candidates": []}) == {"content": ""}
