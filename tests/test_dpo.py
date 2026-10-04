import random

from vi_fc.dpo import build_pairs, mine_on_policy, perturb
from vi_fc.match import actions_match, call_equal, calls_equal
from vi_fc.parse import Action, ToolCall, parse_output
from vi_fc.tools import TOOL_NAMES

STATE = {"speed_kmh": 0, "gear": "P"}


def _rec(gold, scenario="single", seed=None, user="câu"):
    return {"id": "r1", "scenario": scenario, "state": STATE, "history": [], "user": user, "gold": gold,
            "seed": seed or {}}


CLIMATE = {"kind": "call", "calls": [{"name": "set_climate", "arguments": {"temperature": 22}}], "text": ""}


def test_perturb_call_types():
    types = {t for t, _ in perturb(_rec(CLIMATE), random.Random(0))}
    assert {"wrong_tool", "hallucinated_tool", "wrong_value", "wrong_unit", "over_refuse"} <= types
    wu = dict(perturb(_rec(CLIMATE), random.Random(0)))["wrong_unit"]
    assert wu.calls[0].arguments == {"temperature": 220}
    hall = dict(perturb(_rec(CLIMATE), random.Random(0)))["hallucinated_tool"]
    assert hall.calls[0].name not in TOOL_NAMES


def test_perturb_never_equals_gold():
    rng = random.Random(0)
    gold = Action.from_dict(CLIMATE)
    for _ in range(30):
        for _, a in perturb(_rec(CLIMATE), rng):
            assert not actions_match(gold, a)


def test_perturb_multi_and_missing_and_unsafe():
    multi = {"kind": "call", "calls": [{"name": "set_volume", "arguments": {"level": 10}},
                                       {"name": "play_music", "arguments": {"query": "Sơn Tùng"}}]}
    assert "dropped_call" in {t for t, _ in perturb(_rec(multi, "multi"), random.Random(1))}

    ask = {"kind": "ask", "text": "Bạn muốn gọi cho ai?"}
    seed = {"calls": [{"name": "call_contact", "arguments": {}}], "omitted": "name", "omitted_value": "mẹ"}
    (t, a), = perturb(_rec(ask, "missing", seed), random.Random(0))
    assert t == "guess_instead_of_ask" and a.calls == [ToolCall("call_contact", {"name": "mẹ"})]

    ref = {"kind": "refuse", "text": "không an toàn"}
    (t, a), = perturb(_rec(ref, "unsafe", {"calls": [{"name": "unlock_doors", "arguments": {}}]}), random.Random(0))
    assert t == "comply_unsafe" and a.calls[0].name == "unlock_doors"

    (t, a), = perturb(_rec({"kind": "reply", "text": "Chào bạn"}, "chitchat", user="mai trời mưa không"),
                      random.Random(0))
    assert t == "spurious_call" and a.calls[0].arguments == {"query": "mai trời mưa không"}


def test_build_pairs_format():
    pairs = build_pairs([_rec(CLIMATE)], per_record=2)
    assert len(pairs) == 2
    p = pairs[0]
    assert p["messages"][-1]["role"] == "user" and p["messages"][-1]["content"].startswith("<xe>")
    assert p["chosen"]["tool_calls"][0]["function"]["arguments"] == {"temperature": 22}
    assert p["chosen"] != p["rejected"]


def test_mine_on_policy_keeps_only_mistakes():
    recs = [_rec(CLIMATE), _rec(CLIMATE)]
    good = Action("call", [ToolCall("set_climate", {"temperature": 22.0, "power": "on"})])
    bad = parse_output('<tool_call>{"name": "set_climate", "arguments": {"temperature": </tool_call>')
    pairs = mine_on_policy(recs, [good, bad])
    assert len(pairs) == 1 and pairs[0]["type"] == "on_policy"
    assert "<tool_call>" in pairs[0]["rejected"]["content"]
    # dòng output của evaluate.predict cũng dùng được trực tiếp
    row = {"pred": bad.to_dict(), "raw": bad.raw}
    assert mine_on_policy(recs[:1], [row])[0]["rejected"]["content"] == bad.raw
    # raw từ HFBackend còn token kết thúc: cắt đi, template tự thêm lại
    row = {"pred": bad.to_dict(), "raw": bad.raw + "<|im_end|>"}
    assert mine_on_policy(recs[:1], [row])[0]["rejected"]["content"] == bad.raw


def test_match_rules():
    # thứ tự lệnh không quan trọng, giá trị mặc định bỏ qua, chuỗi so không dấu
    a = [ToolCall("set_volume", {"level": 5}), ToolCall("navigate_to", {"destination": "Hồ Gươm"})]
    b = [ToolCall("navigate_to", {"destination": "ho guom"}), ToolCall("set_volume", {"level": 5})]
    assert calls_equal(a, b)
    assert call_equal(ToolCall("set_climate", {}), ToolCall("set_climate", {"power": "on"}))
    assert not call_equal(ToolCall("set_climate", {}), ToolCall("set_climate", {"power": "off"}))
    assert call_equal(ToolCall("lookup_sign", {"code": "P.131a"}), ToolCall("lookup_sign", {"code": "P131a"}))
    assert call_equal(ToolCall("lookup_traffic_law", {"question": "vượt đèn đỏ phạt bao nhiêu"}),
                      ToolCall("lookup_traffic_law", {"question": "xe máy vượt đèn đỏ bị phạt bao nhiêu"}))
    assert not call_equal(ToolCall("navigate_to", {"destination": "Hồ Gươm", "avoid_tolls": False}),
                          ToolCall("navigate_to", {"destination": "Hồ Gươm"}))


def test_fuzzy_place_and_music():
    assert call_equal(ToolCall("play_music", {"query": "Đen Vâu"}), ToolCall("play_music", {"query": "nhạc Đen Vâu"}))
    assert call_equal(ToolCall("navigate_to", {"destination": "sân bay Nội Bài"}),
                      ToolCall("navigate_to", {"destination": "Nội Bài"}))
    assert not call_equal(ToolCall("navigate_to", {"destination": "Hồ Gươm"}), ToolCall("navigate_to", {"destination": "Hồ Tây"}))
    assert not call_equal(ToolCall("navigate_to", {"destination": "sân bay Nội Bài"}),
                          ToolCall("navigate_to", {"destination": "Nội"}))
    assert not call_equal(ToolCall("call_contact", {"name": "anh Tuấn"}), ToolCall("call_contact", {"name": "Tuấn"}))
