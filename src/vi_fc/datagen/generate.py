"""Gọi LLM viết câu người dùng cho từng seed, theo lô.

    python -m vi_fc.datagen.generate --n 4000 --out data/gen/raw.jsonl

Mỗi request gửi ~8 seed: Groq free tier giới hạn số request/phút chặt hơn số token, gom lô thì nhanh hơn
nhiều. Ghi nối tiếp từng lô vào file + bỏ qua seed đã có, để hết quota giữa chừng thì chạy lại là tiếp tục.
Bỏ dấu và lỗi gõ phím do code làm (không nhờ LLM): làm bằng code thì chắc chắn và không đổi nhãn.
"""

import argparse
import json
import logging
import random
import re
from collections.abc import Callable
from pathlib import Path

from ..text import fold
from .seeds import ASK_TEMPLATES, STYLES, Seed, describe_call, gold_action, sample_seed
from .values import ARG_LABEL

log = logging.getLogger(__name__)

NO_DIACRITICS_P = 0.15
TYPO_P = 0.08

SYSTEM = """Bạn giúp tạo dữ liệu huấn luyện cho trợ lý giọng nói trên ô tô điện ở Việt Nam.
Với mỗi mục, hãy viết câu NGƯỜI DÙNG nói với trợ lý (đang ngồi trong xe), đúng yêu cầu và văn phong của mục đó.
Quy tắc:
- Chỉ viết lời người dùng, tiếng Việt tự nhiên như người thật nói, không giải thích.
- Phải nhắc ĐỦ các giá trị được cho (số, tên, địa điểm...), không thêm yêu cầu nào khác.
- Mỗi mục một câu khác nhau, đừng lặp lại cấu trúc câu giữa các mục.
- Số có thể viết bằng chữ số hoặc bằng chữ.
Trả về JSON: {"items": [{"id": "...", "user": "...", ...}]} theo đúng các trường mỗi mục yêu cầu."""


def _slot_label(name: str, slot: str) -> str:
    return ARG_LABEL.get((name, slot), (slot, ""))[0]


def _head(call: dict) -> str:
    return describe_call({"name": call["name"], "arguments": {}})


def seed_instruction(seed: Seed) -> dict:
    sc = seed.scenario
    item = {"id": seed.id, "van_phong": STYLES[seed.style]}
    if sc in ("single", "unsafe"):
        item["yeu_cau"] = f"Người dùng muốn {describe_call(seed.calls[0])}."
    elif sc == "multi":
        a, b = (describe_call(c) for c in seed.calls)
        item["yeu_cau"] = f"Người dùng yêu cầu HAI việc trong MỘT câu: (1) {a}; (2) {b}."
    elif sc == "missing":
        c = seed.calls[0]
        item["yeu_cau"] = (f"Người dùng muốn {describe_call(c)}, nhưng KHÔNG nói "
                           f"{_slot_label(c['name'], seed.omitted)}. Câu phải thiếu đúng thông tin đó.")
    elif sc == "followup":
        c = seed.calls[0]
        val = describe_call({"name": c["name"], "arguments": {seed.omitted: c["arguments"][seed.omitted]}})
        item["yeu_cau"] = (
            f"Hai lượt. 'first': người dùng muốn {_head(c)} nhưng không nói {_slot_label(c['name'], seed.omitted)}. "
            f"Trợ lý hỏi lại: \"{ASK_TEMPLATES[(c['name'], seed.omitted)]}\". "
            f"'user': người dùng trả lời ngắn gọn ({val.split(': ', 1)[-1]})."
        )
        if len(c["arguments"]) > 1:
            others = {k: v for k, v in c["arguments"].items() if k != seed.omitted}
            item["yeu_cau"] += f" Lượt 'first' có nói: {describe_call({'name': c['name'], 'arguments': others})}."
        item["truong"] = "first, user"
    elif sc == "law":
        item["yeu_cau"] = f"Người dùng hỏi về luật giao thông: {seed.topic}. Hỏi tự nhiên như đang lái xe."
    elif sc == "chitchat":
        item["yeu_cau"] = (f"Người dùng nói chuyện ngoài lề: {seed.topic}. Thêm trường 'reply': trợ lý đáp 1-2 câu "
                           "ngắn bằng tiếng Việt, thân thiện, KHÔNG dùng 'xin lỗi', 'không thể', không đặt câu hỏi "
                           "lại. Việc trợ lý trên xe không làm được (đặt vé, đặt bàn...) hoặc cần tin mới (giá vàng, "
                           "thời tiết, bóng đá...) thì nói nhẹ nhàng là mình chưa có thông tin đó / bạn xem trên điện "
                           "thoại nhé. KHÔNG bịa thông tin, KHÔNG nói là đã làm xong.")
        item["truong"] = "user, reply"
    return item


def build_prompt(seeds: list[Seed]) -> str:
    items = [seed_instruction(s) for s in seeds]
    return "Các mục:\n" + json.dumps(items, ensure_ascii=False, indent=1)


def parse_items(text: str) -> dict[str, dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    items = data.get("items", []) if isinstance(data, dict) else data
    return {str(it.get("id")): it for it in items if isinstance(it, dict) and it.get("user")}


def add_typo(text: str, rng: random.Random, protected: set[str]) -> str:
    """Một lỗi gõ (đảo/mất/lặp ký tự) ở một từ ≥4 ký tự, không đụng số và tên riêng trong tham số."""
    words = text.split(" ")
    cand = [i for i, w in enumerate(words)
            if len(w) >= 4 and not any(ch.isdigit() for ch in w) and fold(w).strip(".,!?") not in protected]
    if not cand:
        return text
    i = rng.choice(cand)
    w = words[i]
    j = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif op == "drop":
        w = w[:j] + w[j + 1:]
    else:
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def _fold_args(args: dict) -> dict:
    return {k: fold(v) if isinstance(v, str) else v for k, v in args.items()}


def make_record(seed: Seed, item: dict, rng: random.Random) -> dict | None:
    user_raw = str(item.get("user", "")).strip()
    first_raw = str(item.get("first", "")).strip()
    if not user_raw or (seed.scenario == "followup" and not first_raw):
        return None
    user, first, style = user_raw, first_raw, seed.style
    protected = {fold(w) for c in seed.calls for v in c["arguments"].values() if isinstance(v, str) for w in v.split()}
    if rng.random() < NO_DIACRITICS_P:
        user, first, style = fold(user), fold(first), style + "+no_diacritics"
    if rng.random() < TYPO_P:
        user, style = add_typo(user, rng, protected), style + "+typo"

    gold = gold_action(seed, user, item.get("reply"))
    if "no_diacritics" in style and gold.kind == "call":
        # người dùng gõ không dấu thì tham số chuỗi cũng giữ không dấu: không dạy model tự "đoán dấu" cho tên riêng
        gold.calls = [c._replace(arguments=_fold_args(c.arguments)) for c in gold.calls]
    history = []
    if seed.scenario == "followup":
        c = seed.calls[0]
        history = [{"role": "user", "content": first},
                   {"role": "assistant", "content": ASK_TEMPLATES[(c["name"], seed.omitted)]}]
    return {
        "id": seed.id, "group": seed.group, "scenario": seed.scenario, "style": style,
        "state": seed.state, "history": history, "user": user, "gold": gold.to_dict(),
        "seed": {"calls": seed.calls, "omitted": seed.omitted, "topic": seed.topic, **seed.extra},
        "user_raw": user_raw, "first_raw": first_raw,
    }


def generate(n: int, complete: Callable[[str, str], str], out_path: str | Path, seed: int = 0,
             batch_size: int = 8) -> int:
    """complete(system, user) -> text. Trả về số bản ghi mới ghi được."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open(encoding="utf-8") if line.strip()}
    rng = random.Random(seed)
    # bốc đủ n seed trước rồi mới lọc theo `done`: chạy lại thì seed thứ i vẫn là seed thứ i
    seeds = [sample_seed(rng, i) for i in range(n)]
    todo = [s for s in seeds if s.id not in done]
    aug_rng = random.Random(seed + 1)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), batch_size):
            batch = todo[b : b + batch_size]
            try:
                items = parse_items(complete(SYSTEM, build_prompt(batch)))
            except Exception as e:  # lỗi mạng/quota: bỏ lô này, lần chạy sau làm lại
                log.warning("lô %d lỗi: %s", b // batch_size, e)
                if "mọi model" in str(e):
                    log.warning("hết quota ngày, dừng. Mai chạy lại cell này là sinh tiếp.")
                    break
                continue
            for s in batch:
                rec = make_record(s, items[s.id], aug_rng) if s.id in items else None
                if rec:
                    if getattr(complete, "model", None):
                        rec["gen_model"] = complete.model  # biết câu nào do model nào viết (khi chuyển model)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
            f.flush()
            log.info("%d/%d seed, ghi %d", min(b + batch_size, len(todo)), len(todo), written)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="data/gen/raw.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--provider", default="groq")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from dotenv import load_dotenv

    from ..llm import LLMClient

    load_dotenv()
    client = LLMClient(args.provider)

    def call(system, user):
        r = client.complete(system, user)
        call.model = r.model
        return r.text

    n = generate(args.n, call, args.out, args.seed, args.batch_size)
    print(f"ghi thêm {n} bản ghi vào {args.out}")


if __name__ == "__main__":
    main()
