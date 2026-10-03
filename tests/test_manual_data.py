"""Bộ test viết tay phải luôn khớp schema + luật an toàn hiện tại (sửa luật mà quên sửa nhãn thì CI báo)."""

import json
from pathlib import Path

from vi_fc.inference import guard_action
from vi_fc.parse import Action, ToolCall, classify_text
from vi_fc.safety import check_calls
from vi_fc.state import VehicleState
from vi_fc.tools import validate_call

DATA = Path(__file__).resolve().parents[1] / "data" / "test_manual.jsonl"
UNSAFE_INTENT = {"m38": ("unlock_doors", {}), "m39": ("set_lights", {"mode": "off"}),
                 "m40": ("open_window", {"position": "all"}), "m41": ("unlock_doors", {})}


def _load():
    return [json.loads(x) for x in DATA.open(encoding="utf-8") if x.strip()]


def test_manual_set_is_consistent():
    recs = _load()
    assert len(recs) >= 40
    assert {r["scenario"] for r in recs} == {"single", "multi", "missing", "followup", "unsafe", "law", "chitchat"}
    for r in recs:
        g = Action.from_dict(r["gold"])
        st = VehicleState.from_dict(r["state"])
        for c in g.calls:
            assert validate_call(c.name, c.arguments) == [], r["id"]
        if g.kind == "call":
            assert check_calls(g.calls, st).allowed, r["id"]
        else:
            assert classify_text(g.text) == g.kind, r["id"]


def test_manual_unsafe_labels_match_rules():
    recs = {r["id"]: r for r in _load()}
    for rid, (name, args) in UNSAFE_INTENT.items():
        r = recs[rid]
        guarded = guard_action(Action("call", [ToolCall(name, args)]), r["state"])
        assert guarded.kind == r["gold"]["kind"] and guarded.text == r["gold"]["text"], rid
