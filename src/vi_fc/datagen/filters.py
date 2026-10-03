"""Lọc dữ liệu sinh ra, bỏ trùng, chia train/val/test.

    python -m vi_fc.datagen.filters --raw data/gen/raw.jsonl --out data/gen

Kiểm tra mỗi bản ghi:
- tham số gold khớp schema, nhãn khớp luật an toàn (phòng bug trong sampler),
- text gold (hỏi lại / từ chối / trả lời) được parse.classify_text phân loại đúng loại,
- câu LLM viết có nhắc đủ giá trị trong seed (số, enum, tên...). Đây là lọc quan trọng nhất:
  LLM hay "quên" một tham số hoặc đổi 22 độ thành "mát mát", khi đó nhãn gold (dựng từ seed) thành sai.
Chia theo `group` (nội dung seed) chứ không theo câu: các câu viết từ cùng một yêu cầu luôn nằm cùng một phía,
không thì test bị lộ (model đã thấy gần như y hệt lúc train).
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..parse import Action, classify_text
from ..safety import check_calls
from ..state import VehicleState
from ..text import alnum, char_ngrams, jaccard, mentions_number, normalize
from ..tools import FREE_TEXT_ARGS, validate_call
from .seeds import seed_hash
from .values import ALIASES, NUMBER_ALIASES


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(normalize(phrase))}(?!\w)", normalize(text)) is not None


def mentions_value(text: str, tool: str, arg: str, value) -> bool:
    if (tool, arg) in FREE_TEXT_ARGS:
        if arg == "text":  # nội dung nhắc việc: cho phép diễn đạt lại, chỉ cần nửa số từ xuất hiện
            words = normalize(value).split()
            return sum(w in normalize(text).split() for w in words) >= max(1, len(words) // 2)
        return True
    if (tool, arg) in ALIASES:
        return any(_has_phrase(text, a) for a in ALIASES[(tool, arg)].get(value, []))
    if isinstance(value, bool):
        return True  # bool False không bao giờ được sample
    if isinstance(value, int | float):
        extra = NUMBER_ALIASES.get((tool, arg), {}).get(value, [])
        return mentions_number(text, value) or any(_has_phrase(text, a) for a in extra)
    if tool == "lookup_sign":
        return alnum(value) in alnum(text)
    # tên riêng / địa điểm / bài hát: đủ các âm tiết (đã bỏ dấu), cho phép thêm từ xen giữa
    t = normalize(text).split()
    return all(w in t for w in normalize(value).split())


def check_record(rec: dict) -> list[str]:
    problems = []
    gold = Action.from_dict(rec["gold"])
    state = VehicleState.from_dict(rec["state"])
    sc = rec["scenario"]
    for c in gold.calls:
        problems += validate_call(c.name, c.arguments)
    if gold.kind == "call" and not check_calls(gold.calls, state).allowed:
        problems.append("gold gọi tool nhưng luật an toàn chặn")
    if gold.kind != "call":
        if not gold.text:
            problems.append("gold không có text")
        elif classify_text(gold.text) != gold.kind:
            problems.append(f"text gold bị phân loại thành {classify_text(gold.text)} thay vì {gold.kind}")

    user_raw = rec.get("user_raw") or rec["user"]
    if len(user_raw) > 300:
        problems.append("câu quá dài")
    seed = rec.get("seed", {})
    if sc in ("single", "multi", "unsafe", "missing", "followup"):
        for c in seed.get("calls", []):
            for k, v in c["arguments"].items():
                # followup: giá trị bổ sung nằm ở lượt 2, phần còn lại ở lượt 1
                where = user_raw + " " + rec.get("first_raw", "") if sc == "followup" else user_raw
                if not mentions_value(where, c["name"], k, v):
                    problems.append(f"câu không nhắc {c['name']}.{k}={v!r}")
    if sc == "missing":
        ov = seed.get("omitted_value")
        if isinstance(ov, int | float) and not isinstance(ov, bool) and re.search(r"\d", user_raw):
            # thiếu slot số mà câu lại có số -> nhiều khả năng LLM vẫn nói giá trị
            problems.append("kịch bản thiếu slot số nhưng câu có chữ số")
        elif isinstance(ov, str) and mentions_value(user_raw, seed["calls"][0]["name"], seed["omitted"], ov):
            problems.append("kịch bản thiếu slot nhưng câu vẫn nhắc giá trị")
    if sc == "followup" and not rec.get("history"):
        problems.append("followup thiếu history")
    return problems


def _dedupe_text(rec: dict) -> str:
    hist = " ".join(m["content"] for m in rec.get("history", []) if m["role"] == "user")
    return f"{hist} {rec['user']}".strip()


def dedupe(records: list[dict], threshold: float = 0.85) -> tuple[list[dict], list[dict]]:
    """Bỏ câu gần trùng (Jaccard 3-gram ký tự > threshold, so trên text đã bỏ dấu).
    Chỉ so trong cùng nhóm nhãn (loại + tên tool): câu giống nhau nhưng khác state, khác nhãn là dữ liệu tương
    phản có ích, không được xoá. Gom nhóm cũng giảm số cặp phải so (O(n^2) trong từng nhóm)."""
    buckets: dict[str, list[tuple[set, dict]]] = defaultdict(list)
    kept, dropped = [], []
    for r in records:
        g = r["gold"]
        key = g["kind"] + ":" + ",".join(sorted(c["name"] for c in g.get("calls", [])))
        grams = char_ngrams(_dedupe_text(r))
        if any(abs(len(grams) - len(o)) <= len(grams) * 0.3 and jaccard(grams, o) > threshold
               for o, _ in buckets[key]):
            dropped.append(r)
            continue
        buckets[key].append((grams, r))
        kept.append(r)
    return kept, dropped


def split_by_group(records: list[dict], val: float = 0.05, test: float = 0.10) -> dict[str, list[dict]]:
    """Chia ổn định theo hash của group: thêm dữ liệu mới không làm xáo các bản ghi cũ sang split khác."""
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        h = seed_hash(r["group"])
        out["test" if h < test else "val" if h < test + val else "train"].append(r)
    return out


def stats(records: list[dict]) -> dict:
    tools = Counter(c["name"] for r in records for c in r["gold"].get("calls", []))
    return {
        "n": len(records),
        "scenario": dict(Counter(r["scenario"] for r in records).most_common()),
        "kind": dict(Counter(r["gold"]["kind"] for r in records).most_common()),
        "style": dict(Counter(r["style"] for r in records).most_common()),
        "tools": dict(tools.most_common()),
    }


def build(raw_path: str | Path, out_dir: str | Path, threshold: float = 0.85) -> dict:
    raw = [json.loads(line) for line in Path(raw_path).open(encoding="utf-8") if line.strip()]
    ok, reasons = [], Counter()
    for r in raw:
        p = check_record(r)
        if p:
            reasons[p[0].split("=")[0]] += 1
        else:
            ok.append(r)
    kept, dropped = dedupe(ok, threshold)
    splits = split_by_group(kept)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "raw": len(raw), "invalid": len(raw) - len(ok), "duplicates": len(dropped),
        "invalid_reasons": dict(reasons.most_common(20)),
        "splits": {k: stats(v) for k, v in splits.items()},
    }
    (out_dir / "stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/gen/raw.jsonl")
    ap.add_argument("--out", default="data/gen")
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()
    rep = build(args.raw, args.out, args.threshold)
    print(json.dumps({k: v for k, v in rep.items() if k != "splits"}, ensure_ascii=False, indent=1))
    for k, v in rep["splits"].items():
        print(k, v["n"], v["kind"])


if __name__ == "__main__":
    main()
