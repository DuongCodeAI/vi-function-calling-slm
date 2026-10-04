import json
import random

from vi_fc.datagen.filters import build, check_record, dedupe, mentions_value, split_by_group
from vi_fc.datagen.generate import add_typo, build_prompt, generate, make_record, parse_items
from vi_fc.datagen.seeds import ASK_TEMPLATES, Seed, check_seed_label, describe_call, gold_action, sample_seed
from vi_fc.parse import classify_text
from vi_fc.tools import validate_call

MOVING = {"speed_kmh": 60, "gear": "D", "is_night": True}
STOPPED = {"speed_kmh": 0, "gear": "P"}


def test_gold_single_and_contrast_by_state():
    calls = [{"name": "unlock_doors", "arguments": {}}]
    g = gold_action(Seed("a", "single", calls, STOPPED, "standard"), "mở khoá cửa")
    assert g.kind == "call" and g.calls[0].name == "unlock_doors"
    g = gold_action(Seed("b", "unsafe", calls, MOVING, "standard"), "mở khoá cửa")
    assert g.kind == "refuse" and "60 km/h" in g.text


def test_gold_other_scenarios():
    s = Seed("c", "missing", [{"name": "call_contact", "arguments": {}}], STOPPED, "standard", omitted="name")
    assert gold_action(s, "gọi điện đi").text == ASK_TEMPLATES[("call_contact", "name")]
    s = Seed("d", "law", [{"name": "lookup_traffic_law", "arguments": {}}], STOPPED, "standard", topic="x")
    g = gold_action(s, "vượt đèn đỏ phạt bao nhiêu ")
    assert g.calls[0].arguments == {"question": "vượt đèn đỏ phạt bao nhiêu"}
    s = Seed("e", "chitchat", [], STOPPED, "standard", topic="x")
    assert gold_action(s, "chào", reply="Chào bạn!").kind == "reply"
    win = [{"name": "open_window", "arguments": {"position": "all"}}]
    assert gold_action(Seed("f", "unsafe", win, {"speed_kmh": 110}, "standard"), "x").kind == "ask"


def test_sampled_seeds_give_valid_labels():
    rng = random.Random(1)
    seen = set()
    for i in range(800):
        s = sample_seed(rng, i)
        seen.add(s.scenario)
        assert check_seed_label(s)
        g = gold_action(s, "câu hỏi", reply="Chúc bạn lái xe vui vẻ.")
        for c in g.calls:
            assert validate_call(c.name, c.arguments) == [], (s, c)
        if s.scenario in ("single", "multi", "followup", "law"):
            assert g.kind == "call"
        if s.scenario == "unsafe":
            assert g.kind in ("refuse", "ask")
        if g.kind != "call":
            assert classify_text(g.text) == g.kind, g.text
    assert seen == {"single", "multi", "missing", "followup", "unsafe", "law", "chitchat"}


def test_describe_call():
    assert describe_call({"name": "set_climate", "arguments": {"temperature": 22, "zone": "passenger"}}) == \
        "chỉnh điều hoà: nhiệt độ 22 độ, ghế phụ"
    assert describe_call({"name": "open_window", "arguments": {"position": "driver", "percent": 0}}) == \
        "đóng cửa sổ: cửa sổ ghế lái"


def test_mentions_value():
    assert mentions_value("cho máy lạnh hai mươi hai độ nha", "set_climate", "temperature", 22)
    assert not mentions_value("cho máy lạnh mát xíu", "set_climate", "temperature", 22)
    assert mentions_value("bật AC bên phụ", "set_climate", "zone", "passenger")
    assert mentions_value("chi duong toi ho guom", "navigate_to", "destination", "Hồ Gươm")
    assert not mentions_value("chỉ đường tới hồ tây", "navigate_to", "destination", "Hồ Gươm")
    assert mentions_value("biển p 131a là gì", "lookup_sign", "code", "P.131a")
    assert mentions_value("nửa tiếng nữa nhắc anh", "set_reminder", "minutes", 30)
    assert mentions_value("nhắc anh mua sữa nhé", "set_reminder", "text", "mua sữa cho con")


def _rec(user, seed_calls, gold, scenario="single", state=STOPPED, **kw):
    return {"id": "x", "group": "g", "scenario": scenario, "style": "standard", "state": state, "history": [],
            "user": user, "user_raw": user, "gold": gold, "seed": {"calls": seed_calls, **kw}}


def test_check_record():
    calls = [{"name": "set_volume", "arguments": {"level": 12}}]
    gold = {"kind": "call", "calls": calls, "text": ""}
    assert check_record(_rec("vặn loa lên 12 nhé", calls, gold)) == []
    assert any("không nhắc" in p for p in check_record(_rec("vặn loa to lên", calls, gold)))
    bad = {"kind": "call", "calls": [{"name": "set_volume", "arguments": {"level": 99}}], "text": ""}
    assert check_record(_rec("vặn loa lên 12 nhé", calls, bad))
    unlock = [{"name": "unlock_doors", "arguments": {}}]
    assert check_record(_rec("mở khoá", unlock, {"kind": "call", "calls": unlock}, state=MOVING))
    # thiếu slot số mà câu vẫn có số
    miss = _rec("chỉnh âm lượng 10", [{"name": "set_volume", "arguments": {}}],
                {"kind": "ask", "text": "Bạn muốn chỉnh âm lượng mức bao nhiêu, từ 0 đến 30?"},
                scenario="missing", omitted="level", omitted_value=10)
    assert check_record(miss)


def test_dedupe_keeps_state_contrast():
    a = {"user": "mở khoá cửa giúp mình", "gold": {"kind": "call", "calls": [{"name": "unlock_doors"}]}}
    b = {"user": "Mở khoá cửa giúp mình!", "gold": {"kind": "call", "calls": [{"name": "unlock_doors"}]}}
    c = {"user": "mở khoá cửa giúp mình", "gold": {"kind": "refuse", "calls": []}}
    d = {"user": "mo khoa cua giup minh", "gold": {"kind": "call", "calls": [{"name": "unlock_doors"}]}}
    kept, dropped = dedupe([a, b, c, d])
    assert kept == [a, c] and dropped == [b, d]


def test_split_by_group_no_leak():
    recs = [{"group": f"g{i % 50}", "i": i} for i in range(500)]
    s1, s2 = split_by_group(recs), split_by_group(list(reversed(recs)))
    groups = {k: {r["group"] for r in v} for k, v in s1.items()}
    assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["val"])
    assert {k: len(v) for k, v in s1.items()} == {k: len(v) for k, v in s2.items()}


def test_add_typo_protects_numbers_and_names():
    rng = random.Random(0)
    for _ in range(50):
        out = add_typo("chỉnh nhiệt độ 22 độ cho Phương", rng, {"phuong"})
        assert "22" in out and "Phương" in out


def test_parse_items_tolerant():
    assert parse_items('{"items": [{"id": "s1", "user": "a"}, {"id": "s2"}]}') == {"s1": {"id": "s1", "user": "a"}}
    assert "s1" in parse_items('Đây:\n{"items": [{"id": "s1", "user": "a"}]}\n')
    assert parse_items("lỗi") == {}


def _fake_llm(n, seed=0):
    """LLM giả: viết câu bằng chính mô tả seed -> luôn nhắc đủ giá trị."""
    rng = random.Random(seed)
    seeds = {s.id: s for s in (sample_seed(rng, i) for i in range(n))}

    def complete(system, user):
        items = json.loads(user.split("\n", 1)[1])
        out = []
        for it in items:
            s = seeds[it["id"]]
            if s.scenario == "chitchat":
                out.append({"id": s.id, "user": f"ê {s.topic}", "reply": "Chúc bạn một ngày vui vẻ."})
            elif s.scenario == "law":
                out.append({"id": s.id, "user": f"cho hỏi {s.topic}"})
            elif s.scenario == "followup":
                c = s.calls[0]
                first = {k: v for k, v in c["arguments"].items() if k != s.omitted}
                out.append({"id": s.id, "first": describe_call({"name": c["name"], "arguments": first}),
                            "user": describe_call({"name": c["name"], "arguments": {s.omitted: c["arguments"][s.omitted]}})})
            else:
                out.append({"id": s.id, "user": " và ".join(describe_call(c) for c in s.calls)})
        return json.dumps({"items": out}, ensure_ascii=False)

    return complete


def test_generate_and_build_end_to_end(tmp_path):
    raw = tmp_path / "raw.jsonl"
    n = generate(120, _fake_llm(120), raw, seed=0, batch_size=8)
    assert n == 120
    # chạy lại: không sinh trùng seed đã có
    assert generate(120, _fake_llm(120), raw, seed=0) == 0
    rep = build(raw, tmp_path / "out")
    total = sum(v["n"] for v in rep["splits"].values())
    assert total + rep["invalid"] + rep["duplicates"] == 120
    assert rep["invalid"] < 25, rep["invalid_reasons"]
    train = [json.loads(x) for x in (tmp_path / "out" / "train.jsonl").open(encoding="utf-8")]
    test = [json.loads(x) for x in (tmp_path / "out" / "test.jsonl").open(encoding="utf-8")]
    assert not ({r["group"] for r in train} & {r["group"] for r in test})
    fu = [r for r in train if r["scenario"] == "followup"]
    if fu:
        assert fu[0]["history"][1]["role"] == "assistant"


def test_make_record_no_diacritics_folds_args():
    s = Seed("s9", "single", [{"name": "navigate_to", "arguments": {"destination": "Hồ Gươm"}}], STOPPED, "standard")

    class AlwaysLow(random.Random):
        def random(self):
            return 0.0

    rec = make_record(s, {"user": "dẫn đường tới Hồ Gươm"}, AlwaysLow())
    assert rec["user"].endswith("ho guom") and "typo" in rec["style"]  # tên riêng không bị gõ sai
    assert rec["gold"]["calls"][0]["arguments"] == {"destination": "ho guom"}
    assert "no_diacritics" in rec["style"] and rec["user_raw"] == "dẫn đường tới Hồ Gươm"
    assert "s9" in build_prompt([s])


def test_text_quality_filters():
    # các lỗi thật gặp ở đợt sinh đầu tiên
    win = [{"name": "open_window", "arguments": {"percent": 50}}]
    leak = _rec("Mở cửa sổ 50% đi, nhưng chưa nói cửa nào.", win, {"kind": "ask", "text": "Bạn muốn mở cửa sổ nào?"},
                scenario="missing", omitted="position", omitted_value="driver")
    assert "câu chép lại đề bài" in check_record(leak)
    vol = [{"name": "set_volume", "arguments": {"level": 11}}]
    eng = _rec("Can you set the volume to 11 for me please", vol, {"kind": "call", "calls": vol})
    assert "câu gần như toàn tiếng Anh" in check_record(eng)
    ok = _rec("Bật volume lên 11 đi", vol, {"kind": "call", "calls": vol})
    assert check_record(ok) == []
    fake = _rec("Đặt vé máy bay cho mình nhé", [], {"kind": "reply", "text": "Đã xong, vé của bạn đã được đặt."},
                scenario="chitchat")
    assert "chitchat bịa là đã làm / bịa thông tin" in check_record(fake)
    en_reply = _rec("Bạn biết hát không?", [], {"kind": "reply", "text": "I can sing a little!"}, scenario="chitchat")
    assert "chitchat trả lời không phải tiếng Việt" in check_record(en_reply)


def test_chitchat_reply_with_numbers_is_fabricated():
    rec = _rec("Thời tiết ngày mai thế nào?", [], {"kind": "reply", "text": "Ngày mai trời nắng nhẹ, khoảng 28 độ C."},
               scenario="chitchat")
    assert "chitchat bịa số liệu" in check_record(rec)
