"""Dựng cặp preference cho DPO: chosen = gold, rejected = lỗi điển hình của model nhỏ.

    python -m vi_fc.dpo --data data/gen/train.jsonl --out data/gen/dpo_train.jsonl

Hai nguồn rejected:
1. nhiễu có chủ đích từ gold (rẻ, phủ đều các kiểu lỗi): sai tool, bịa tên tool, sai giá trị, sai đơn vị
   (220 thay vì 22 độ), thiếu tham số, bỏ sót lệnh thứ hai, đoán bừa thay vì hỏi lại, làm theo yêu cầu
   không an toàn, từ chối thừa, gọi tool cho câu chuyện phiếm.
2. lỗi thật của model SFT (on-policy): chạy SFT trên tập train, câu nào sai thì lấy output đó làm rejected.
   Loại này sát với phân phối lỗi thật hơn, nhưng chỉ có sau khi SFT xong (notebook 03).
"""

import argparse
import json
import random
from pathlib import Path

from .datagen.values import CONTACTS, DESTINATIONS, MUSIC, REMINDERS, SIGN_CODES
from .match import actions_match
from .parse import Action, ToolCall
from .prompt import assistant_message, record_messages
from .tools import FREE_TEXT_ARGS, TOOLS_BY_NAME

FAKE_TOOLS = ["set_temperature", "turn_on_ac", "open_door", "call_phone", "play_song", "set_ac", "navigate",
              "unlock_car", "check_battery", "search_law", "set_fan", "volume_up"]
CHITCHAT_FAKE_TOOLS = ["get_weather", "search_web", "book_ticket", "tell_joke"]
CONFUSABLE = {
    "set_climate": ["set_fan_speed"], "set_fan_speed": ["set_climate"], "set_volume": ["play_music"],
    "play_music": ["set_volume"], "lock_doors": ["unlock_doors"], "unlock_doors": ["lock_doors", "open_window"],
    "navigate_to": ["find_charging_station"], "find_charging_station": ["navigate_to"],
    "lookup_traffic_law": ["lookup_sign"], "lookup_sign": ["lookup_traffic_law"], "open_window": ["unlock_doors"],
    "call_contact": ["read_messages"], "read_messages": ["call_contact"], "set_lights": ["set_drive_mode"],
    "set_drive_mode": ["set_lights"], "get_vehicle_status": ["find_charging_station"],
    "set_reminder": ["read_messages"],
}
STRING_POOLS = {"destination": DESTINATIONS, "name": CONTACTS, "query": MUSIC, "text": REMINDERS, "code": SIGN_CODES}
OVER_REFUSE = "Xin lỗi, mình không thể làm việc này lúc xe đang chạy."


def _other_value(name: str, key: str, value, rng: random.Random):
    spec = TOOLS_BY_NAME[name]["parameters"]["properties"][key]
    if "enum" in spec:
        return rng.choice([v for v in spec["enum"] if v != value])
    if spec["type"] == "boolean":
        return not value
    if spec["type"] in ("integer", "number"):
        lo, hi = spec.get("minimum", 0), spec.get("maximum", 100)
        cand = [v for v in range(int(lo), int(hi) + 1) if v != value]
        return rng.choice(cand)
    pool = [v for v in STRING_POOLS.get(key, []) if v != value]
    return rng.choice(pool) if pool else None


def _wrong_unit(name: str, args: dict) -> dict | None:
    for key in ("temperature", "level", "percent"):
        if isinstance(args.get(key), int | float) and args[key] > 0:
            return {**args, key: args[key] * 10}
    if isinstance(args.get("minutes"), int) and args["minutes"] >= 60:
        return {**args, "minutes": args["minutes"] // 60}  # đổi ra giờ nhưng vẫn để trường phút
    return None


def perturb(rec: dict, rng: random.Random) -> list[tuple[str, Action]]:
    """Mọi biến thể rejected có thể có cho một bản ghi: [(loại lỗi, action sai)]."""
    gold = Action.from_dict(rec["gold"])
    seed = rec.get("seed", {})
    out: list[tuple[str, Action]] = []
    if gold.kind == "call":
        i = rng.randrange(len(gold.calls))
        c = gold.calls[i]

        def with_call(new: ToolCall) -> Action:
            calls = list(gold.calls)
            calls[i] = new
            return Action("call", calls)

        if c.name in CONFUSABLE:
            out.append(("wrong_tool", with_call(ToolCall(rng.choice(CONFUSABLE[c.name]), c.arguments))))
        out.append(("hallucinated_tool", with_call(ToolCall(rng.choice(FAKE_TOOLS), c.arguments))))
        keys = [k for k in c.arguments if (c.name, k) not in FREE_TEXT_ARGS]
        if keys:
            k = rng.choice(keys)
            v = _other_value(c.name, k, c.arguments[k], rng)
            if v is not None:
                out.append(("wrong_value", with_call(ToolCall(c.name, {**c.arguments, k: v}))))
        if (wu := _wrong_unit(c.name, c.arguments)) is not None:
            out.append(("wrong_unit", with_call(ToolCall(c.name, wu))))
        req = TOOLS_BY_NAME[c.name]["parameters"]["required"]
        if req:
            k = rng.choice(req)
            out.append(("missing_arg", with_call(ToolCall(c.name, {a: v for a, v in c.arguments.items() if a != k}))))
        if len(gold.calls) > 1:
            out.append(("dropped_call", Action("call", gold.calls[:1])))
        if rec.get("scenario") != "law":
            out.append(("over_refuse", Action("refuse", text=OVER_REFUSE)))
    elif gold.kind == "ask":
        calls = seed.get("calls") or []
        if calls and seed.get("omitted"):
            # đoán bừa slot bị thiếu thay vì hỏi lại
            c = calls[0]
            val = seed.get("omitted_value")
            if val is None:
                val = _other_value(c["name"], seed["omitted"], None, rng)
            out.append(("guess_instead_of_ask", Action("call", [ToolCall(c["name"], {**c["arguments"],
                                                                                    seed["omitted"]: val})])))
        elif calls:  # hỏi xác nhận (mở cửa sổ khi chạy nhanh) -> làm luôn không hỏi
            out.append(("skip_confirm", Action("call", [ToolCall(c["name"], c["arguments"]) for c in calls])))
    elif gold.kind == "refuse":
        calls = seed.get("calls") or []
        if calls:
            out.append(("comply_unsafe", Action("call", [ToolCall(c["name"], c["arguments"]) for c in calls])))
    elif gold.kind == "reply":
        fake = rng.choice(CHITCHAT_FAKE_TOOLS)
        out.append(("spurious_call", Action("call", [ToolCall(fake, {"query": rec["user"]})])))
    # bỏ biến thể vô tình trùng gold (vd đổi tool nhưng so khớp vẫn đúng)
    return [(t, a) for t, a in out if not actions_match(gold, a)]


def _pair(rec: dict, rejected: Action, kind: str) -> dict:
    if rejected.raw and not rejected.valid_format:
        rej = {"role": "assistant", "content": rejected.raw}  # output hỏng format: giữ nguyên chuỗi
    else:
        rej = assistant_message(rejected)
    return {
        "id": rec["id"], "type": kind,
        "messages": record_messages(rec, with_gold=False),
        "chosen": assistant_message(Action.from_dict(rec["gold"])),
        "rejected": rej,
    }


def build_pairs(records: list[dict], seed: int = 0, per_record: int = 1) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for rec in records:
        cands = perturb(rec, rng)
        for kind, act in rng.sample(cands, min(per_record, len(cands))):
            pairs.append(_pair(rec, act, kind))
    return pairs


def mine_on_policy(records: list[dict], predictions: list[Action | dict]) -> list[dict]:
    """predictions[i] là output của model SFT cho records[i]. Giữ những câu model làm sai làm rejected.
    Output không parse được thì giữ nguyên chuỗi raw: dạy model tránh cả lỗi format."""
    pairs = []
    for rec, pred in zip(records, predictions, strict=True):
        pred = pred if isinstance(pred, Action) else Action.from_dict(pred)
        if not actions_match(Action.from_dict(rec["gold"]), pred) or not pred.valid_format:
            pairs.append(_pair(rec, pred, "on_policy"))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gen/train.jsonl")
    ap.add_argument("--out", default="data/gen/dpo_train.jsonl")
    ap.add_argument("--per-record", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    recs = [json.loads(x) for x in Path(args.data).open(encoding="utf-8") if x.strip()]
    pairs = build_pairs(recs, args.seed, args.per_record)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter

    print(len(pairs), "cặp", dict(Counter(p["type"] for p in pairs).most_common()))


if __name__ == "__main__":
    main()
