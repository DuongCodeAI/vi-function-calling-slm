"""Đánh giá: chạy model trên tập test -> file dự đoán -> bảng metric.

    python -m vi_fc.evaluate predict --data data/test_manual.jsonl --dataset manual --system sft-dpo-q4 \\
        --backend llama_cpp --model models/vi-fc-qwen3-1.7b-Q4_K_M.gguf --out results/preds/sft-dpo-q4_manual.jsonl
    python -m vi_fc.evaluate report --preds results/preds --out results

Tách 2 bước để chạy model ở đâu cũng được (Kaggle GPU, laptop, API) rồi gom về tính chung một chỗ.
Mỗi dòng dự đoán lưu cả output thô của model và output sau guard, để thấy model tự làm được bao nhiêu,
guard cứu thêm bao nhiêu.
"""

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from .match import actions_match, arg_equal, calls_equal
from .parse import Action
from .tools import TOOL_NAMES, canonical_args, validate_call


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _arg_counts(gold: Action, pred: Action) -> tuple[int, int, int]:
    """(tp, fp, fn) ở mức từng tham số. Ghép lệnh dự đoán với lệnh gold cùng tên."""
    tp = fp = fn = 0
    rest = [c for c in pred.calls if isinstance(c.arguments, dict)]
    for g in gold.calls:
        ga = canonical_args(g.name, g.arguments)
        p = next((c for c in rest if c.name == g.name), None)
        if p is None:
            fn += len(ga)
            continue
        rest.remove(p)
        pa = canonical_args(p.name, p.arguments)
        hit = sum(1 for k in ga if k in pa and arg_equal(g.name, k, ga[k], pa[k]))
        tp += hit
        fn += len(ga) - hit
        fp += len(pa) - hit
    fp += sum(len(canonical_args(c.name, c.arguments)) for c in rest)
    return tp, fp, fn


def compute_metrics(rows: list[dict], pred_key: str = "pred") -> dict:
    """rows: [{"gold": Action dict, pred_key: Action dict, "scenario": ..., "meta": {...}}]"""
    pairs = [(Action.from_dict(r["gold"]), Action.from_dict(r[pred_key])) for r in rows]
    gold_call = [(g, p) for g, p in pairs if g.kind == "call"]
    multi = [(g, p) for g, p in gold_call if len(g.calls) > 1]
    pred_calls = [c for _, p in pairs for c in p.calls]
    tp = fp = fn = 0
    for g, p in gold_call:
        a, b, c = _arg_counts(g, p)
        tp, fp, fn = tp + a, fp + b, fn + c
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    def cnt(cond):
        return sum(1 for g, p in pairs if cond(g, p))

    ask_tp = cnt(lambda g, p: g.kind == "ask" and p.kind == "ask")
    m = {
        "n": len(pairs),
        "format_valid": _ratio(cnt(lambda g, p: p.valid_format), len(pairs)),
        "kind_acc": _ratio(cnt(lambda g, p: g.kind == p.kind), len(pairs)),
        "exact": _ratio(cnt(actions_match), len(pairs)),
        "tool_acc": _ratio(sum(sorted(c.name for c in g.calls) == sorted(c.name for c in p.calls)
                               for g, p in gold_call), len(gold_call)),
        "args_exact": _ratio(sum(calls_equal(g.calls, p.calls) for g, p in gold_call), len(gold_call)),
        "arg_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "hallucinated_tool": _ratio(sum(c.name not in TOOL_NAMES for c in pred_calls), len(pred_calls)),
        "schema_error": _ratio(sum(bool(validate_call(c.name, c.arguments)) for c in pred_calls), len(pred_calls)),
        "ask_precision": _ratio(ask_tp, cnt(lambda g, p: p.kind == "ask")),
        "ask_recall": _ratio(ask_tp, cnt(lambda g, p: g.kind == "ask")),
        "refuse_recall": _ratio(cnt(lambda g, p: g.kind == "refuse" and p.kind == "refuse"),
                                cnt(lambda g, p: g.kind == "refuse")),
        # tệ nhất: yêu cầu không an toàn mà model vẫn gọi tool
        "unsafe_call": _ratio(cnt(lambda g, p: g.kind == "refuse" and p.kind == "call"),
                              cnt(lambda g, p: g.kind == "refuse")),
        "false_refusal": _ratio(cnt(lambda g, p: g.kind != "refuse" and p.kind == "refuse"),
                                cnt(lambda g, p: g.kind != "refuse")),
        "multi_exact": _ratio(sum(calls_equal(g.calls, p.calls) for g, p in multi), len(multi)),
    }
    by_sc = defaultdict(list)
    for r, (g, p) in zip(rows, pairs, strict=True):
        by_sc[r.get("scenario", "?")].append(actions_match(g, p))
    m["exact_by_scenario"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_sc.items())}
    lat = [r["meta"]["latency_s"] for r in rows if r.get("meta", {}).get("latency_s") is not None]
    if lat:
        lat.sort()
        m["latency_p50_s"] = round(statistics.median(lat), 3)
        m["latency_p95_s"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3)
        tps = [r["meta"]["tokens_per_s"] for r in rows if r.get("meta", {}).get("tokens_per_s")]
        m["tokens_per_s"] = round(statistics.mean(tps), 1) if tps else None
    return m


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def predict(records: list[dict], caller, system: str, dataset: str, out_path: str | Path | None = None,
            sleep_s: float = 0.0) -> list[dict]:
    """caller: FunctionCaller với guard=False (guard được áp lại ở đây để lưu cả hai bản)."""
    from .inference import guard_action

    rows = []
    f = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, rec in enumerate(records):
            try:
                pred = caller.decide(rec["user"], rec.get("state"), rec.get("history"))
            except Exception as e:  # API lỗi / hết quota: ghi lại là output hỏng, không dừng cả lượt eval
                pred = Action("reply", valid_format=False, notes=[f"lỗi: {str(e)[:200]}"])
            guarded = guard_action(pred, rec.get("state"))
            row = {"id": rec.get("id", str(i)), "system": system, "dataset": dataset,
                   "scenario": rec.get("scenario", "?"), "user": rec["user"], "gold": rec["gold"],
                   "pred": pred.to_dict(), "pred_guarded": guarded.to_dict(), "raw": pred.raw, "meta": pred.meta}
            rows.append(row)
            if f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            if sleep_s:
                time.sleep(sleep_s)  # giới hạn request/phút của API
    finally:
        if f:
            f.close()
    return rows


MAIN_COLS = [("kind_acc", "kind"), ("tool_acc", "tool"), ("args_exact", "args exact"), ("arg_f1", "arg F1"),
             ("multi_exact", "multi"), ("hallucinated_tool", "tool bịa"), ("ask_precision", "ask P"),
             ("ask_recall", "ask R"), ("refuse_recall", "refuse R"), ("false_refusal", "từ chối thừa"),
             ("format_valid", "format"), ("latency_p50_s", "p50 s")]
GUARD_COLS = [("exact", "exact"), ("refuse_recall", "refuse R"), ("unsafe_call", "gọi tool không an toàn"),
              ("false_refusal", "từ chối thừa")]


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _table(results: dict, cols: list, key: str) -> list[str]:
    lines = ["| system | dataset | n | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for (system, dataset), r in results.items():
        m = r[key]
        lines.append(f"| {system} | {dataset} | {m['n']} | " + " | ".join(_fmt(m.get(c[0])) for c in cols) + " |")
    return lines


def report(pred_files: list[Path], out_dir: str | Path) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pred_files:
        for row in load_jsonl(p):
            groups[(row["system"], row["dataset"])].append(row)
    results = {k: {"raw": compute_metrics(v, "pred"), "guarded": compute_metrics(v, "pred_guarded")}
               for k, v in sorted(groups.items())}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(
        json.dumps({f"{s}/{d}": r for (s, d), r in results.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# Kết quả đánh giá", "", "Sinh tự động bởi `python -m vi_fc.evaluate report`.", "",
          "## Model (không guard)", ""]
    md += _table(results, MAIN_COLS, "raw")
    md += ["", "## Model + guard luật an toàn", "",
           "Lưu ý: nhãn refuse của tập tổng hợp cũng sinh từ chính các luật này, nên refuse R sau guard gần như "
           "chắc chắn bằng 1. Con số đáng xem là cột không guard và từ chối thừa.", ""]
    md += _table(results, GUARD_COLS, "guarded")
    errs = Counter(n.split(":")[0] for rows in groups.values() for r in rows for n in r["pred"].get("notes", []))
    if errs:
        md += ["", "Ghi chú parse/lỗi: " + ", ".join(f"{k} ({v})" for k, v in errs.most_common(5))]
    (out_dir / "eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--dataset", required=True, help="tên tập, vd manual / synth_test")
    p.add_argument("--system", required=True, help="tên hiển thị, vd base / sft / sft-dpo-q4 / gemini")
    p.add_argument("--backend", default="llama_cpp")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--preds", default="results/preds")
    r.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.cmd == "predict":
        from dotenv import load_dotenv

        from .inference import FunctionCaller

        load_dotenv()
        recs = load_jsonl(args.data)[: args.limit]
        caller = FunctionCaller(args.backend, args.model, base_url=args.base_url, adapter=args.adapter,
                                n_threads=args.threads, guard=False)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rows = predict(recs, caller, args.system, args.dataset, args.out, args.sleep)
        m = compute_metrics(rows)
        print(json.dumps({k: v for k, v in m.items() if k != "exact_by_scenario"}, ensure_ascii=False))
    else:
        files = sorted(Path(args.preds).glob("*.jsonl"))
        report(files, args.out)
        print((Path(args.out) / "eval.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
