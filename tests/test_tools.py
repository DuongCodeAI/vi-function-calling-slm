from vi_fc.text import fold, mentions_number, normalize, number_words, token_f1
from vi_fc.tools import TOOL_NAMES, openai_tools, validate, validate_call


def test_tool_list():
    assert len(TOOL_NAMES) == 17
    assert len(set(TOOL_NAMES)) == len(TOOL_NAMES)
    t = openai_tools(["set_volume"])[0]
    assert t["type"] == "function" and t["function"]["name"] == "set_volume"


def test_openai_tools_is_a_copy():
    openai_tools()[0]["function"]["name"] = "hacked"
    assert "set_climate" in TOOL_NAMES and openai_tools()[0]["function"]["name"] == "set_climate"


def test_valid_calls():
    assert validate_call("set_climate", {"temperature": 22, "zone": "driver"}) == []
    assert validate_call("set_climate", {"temperature": 22.5}) == []
    assert validate_call("set_climate", {}) == []
    assert validate_call("lock_doors", {}) == []
    assert validate_call("open_window", {"position": "rear_left", "percent": 0}) == []


def test_invalid_calls():
    assert validate_call("turn_on_ac", {}) == ["tool không tồn tại: turn_on_ac"]
    # sai đơn vị: 220 thay vì 22 độ
    assert any("> 30" in e for e in validate_call("set_climate", {"temperature": 220}))
    assert any("thiếu 'level'" in e for e in validate_call("set_volume", {}))
    assert any("không thuộc" in e for e in validate_call("set_lights", {"mode": "fog"}))
    assert any("tham số lạ" in e for e in validate_call("set_climate", {"temperature": 22, "unit": "C"}))
    assert any("chuỗi rỗng" in e for e in validate_call("navigate_to", {"destination": "  "}))
    assert validate_call("set_volume", "10") == ["set_volume: cần kiểu object, nhận str"]


def test_bool_is_not_integer():
    assert validate(True, {"type": "integer"})
    assert validate(1, {"type": "boolean"})
    assert validate("22", {"type": "number"})


def test_fold_and_normalize():
    assert fold("Đường Lê Lợi") == "duong le loi"
    assert normalize("Bật AC lên 22 độ, nhé!!") == "bat ac len 22 do nhe"


def test_number_words():
    assert "hai muoi hai" in number_words(22) and "hai hai" in number_words(22)
    assert "hai lam" in number_words(25) and "muoi lam" in number_words(15)
    assert "hai tu" in number_words(24)
    assert mentions_number("cho máy lạnh hai mươi hai độ", 22)
    assert mentions_number("bat AC 22 do", 22)
    assert not mentions_number("bat AC 220 do", 22)
    assert mentions_number("để 22,5 độ", 22.5)


def test_token_f1():
    assert token_f1("mua sữa cho con", "mua sua cho con") == 1.0
    assert 0 < token_f1("nhắc mua sữa", "mua sữa cho con") < 1
