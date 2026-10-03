from vi_fc.parse import Action, ToolCall, classify_text, from_openai_message, parse_output
from vi_fc.prompt import (
    assistant_message,
    build_messages,
    record_messages,
    render_chat_template,
    split_prompt_completion,
)
from vi_fc.state import VehicleState
from vi_fc.tools import openai_tools


def test_parse_single_call():
    a = parse_output('<tool_call>\n{"name": "set_climate", "arguments": {"temperature": 22}}\n</tool_call>')
    assert a.kind == "call" and a.valid_format
    assert a.calls == [ToolCall("set_climate", {"temperature": 22})]
    name, args = a.calls[0]  # unpack được như tuple
    assert name == "set_climate"


def test_parse_multi_call_and_think():
    raw = ('<think>\n\n</think>\n\n<tool_call>\n{"name": "set_volume", "arguments": {"level": 10}}\n</tool_call>\n'
           '<tool_call>\n{"name": "play_music", "arguments": {"query": "Sơn Tùng"}}\n</tool_call><|im_end|>')
    a = parse_output(raw)
    assert [c.name for c in a.calls] == ["set_volume", "play_music"]
    assert a.text == ""


def test_parse_bare_json_and_fence():
    a = parse_output('```json\n{"name": "lock_doors", "arguments": {}}\n```')
    assert a.kind == "call" and a.calls[0].name == "lock_doors"
    a = parse_output('[{"name": "lock_doors", "parameters": {}}, {"name": "set_volume", "arguments": "{\\"level\\": 3}"}]')
    assert [c.arguments for c in a.calls] == [{}, {"level": 3}]


def test_parse_truncated_and_broken():
    a = parse_output('<tool_call>\n{"name": "navigate_to", "arguments": {"destination": "Hồ Gươm"}}')
    assert a.kind == "call" and a.calls[0].arguments == {"destination": "Hồ Gươm"}
    a = parse_output('<tool_call>\n{"name": "navigate_to", "arguments": {"destination": </tool_call>')
    assert a.kind == "call" and not a.valid_format and a.calls == []
    a = parse_output('<tool_call>{"name": "lock_doors", "arguments": {}}}\n</tool_call>')  # thừa ngoặc
    assert a.calls == [ToolCall("lock_doors", {})]


def test_parse_text():
    assert parse_output("Bạn muốn đi đến đâu?").kind == "ask"
    assert parse_output("Xe đang chạy 60 km/h, mở khoá cửa lúc này không an toàn.").kind == "refuse"
    assert parse_output("Chào bạn, chúc bạn lái xe vui vẻ!").kind == "reply"
    a = parse_output('{"answer": "ok"}')
    assert not a.valid_format


def test_classify_text_priority():
    assert classify_text("Bạn muốn quạt ở mức mấy, từ 1 đến 7?") == "ask"
    assert classify_text("Bạn muốn nghe bài gì ạ") == "ask"
    assert classify_text("Bạn chắc chắn muốn mở chứ?") == "ask"
    # có "không an toàn" thì vẫn là từ chối dù có dấu hỏi
    assert classify_text("Không an toàn đâu, bạn dừng xe trước được không?") == "refuse"
    assert classify_text("Xin lỗi, mình không thể đặt vé máy bay.") == "refuse"  # giới hạn đã biết
    assert classify_text("Xin lỗi, bạn muốn gọi cho ai?") == "ask"
    assert classify_text("") == "reply"


def test_openai_message():
    msg = {"content": None, "tool_calls": [{"id": "1", "type": "function",
                                            "function": {"name": "set_volume", "arguments": '{"level": 12}'}}]}
    assert from_openai_message(msg).calls == [ToolCall("set_volume", {"level": 12})]
    assert from_openai_message({"content": "Bạn muốn gọi cho ai?"}).kind == "ask"


def test_action_roundtrip():
    a = Action("call", [ToolCall("set_volume", {"level": 5})])
    assert Action.from_dict(a.to_dict()) == a


def test_build_messages_state_in_last_user_turn():
    hist = [{"role": "user", "content": "gọi điện"}, {"role": "assistant", "content": "Bạn muốn gọi cho ai?"}]
    msgs = build_messages("mẹ", VehicleState(speed_kmh=30), hist)
    assert msgs[0]["role"] == "system" and msgs[1]["content"] == "gọi điện"
    assert msgs[-1]["content"].startswith("<xe>tốc độ 30 km/h") and msgs[-1]["content"].endswith("\nmẹ")


def test_assistant_message():
    m = assistant_message(Action("call", [ToolCall("lock_doors", {})]))
    assert m["tool_calls"][0]["function"] == {"name": "lock_doors", "arguments": {}}
    assert assistant_message(Action("ask", text="Đi đâu?")) == {"role": "assistant", "content": "Đi đâu?"}


# template rút gọn theo đúng cú pháp phần tool của Qwen3 (bản đầy đủ nằm trong tokenizer_config / GGUF)
MINI_TEMPLATE = (
    "{%- if tools %}{{- '<|im_start|>system\n' + messages[0].content + '\n<tools>' }}"
    "{%- for tool in tools %}{{- '\n' }}{{- tool | tojson }}{%- endfor %}"
    "{{- '\n</tools><|im_end|>\n' }}{%- endif %}"
    "{%- for message in messages[1:] %}"
    "{%- if message.role == 'user' %}{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<|im_start|>assistant\n' }}"
    "{%- if loop.last %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{{- message.content }}"
    "{%- for tool_call in message.tool_calls or [] %}"
    "{%- if tool_call.function %}{%- set tool_call = tool_call.function %}{%- endif %}"
    "{{- '\n<tool_call>\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' }}"
    "{{- tool_call.arguments | tojson }}{{- '}\n</tool_call>' }}"
    "{%- endfor %}{{- '<|im_end|>\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}"
    "{%- endif %}"
)


def test_render_template_keeps_vietnamese():
    rec = {"user": "bật điều hoà 22 độ", "state": {}, "gold": {"kind": "call", "calls": [
        {"name": "set_climate", "arguments": {"temperature": 22}}]}}
    msgs = record_messages(rec)
    tools = openai_tools(["set_climate"])
    out = render_chat_template(MINI_TEMPLATE, msgs, tools, add_generation_prompt=False)
    assert "Bật/tắt và chỉnh điều hoà" in out and "\\u" not in out
    assert '{"name": "set_climate", "arguments": {"temperature": 22}}' in out

    def apply(m, gen):
        return render_chat_template(MINI_TEMPLATE, m, tools, add_generation_prompt=gen)

    prompt, completion = split_prompt_completion(apply, msgs)
    assert prompt.endswith("</think>\n\n") and completion.startswith("\n<tool_call>")
