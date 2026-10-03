"""So khớp tool call dự đoán với gold (dùng cho eval và để chọn lỗi on-policy cho DPO).

Quy ước:
- thứ tự các lệnh trong một câu không quan trọng ("bật điều hoà và mở nhạc" gọi cái nào trước cũng được),
- tham số bằng giá trị mặc định coi như không có ({"power": "on"} == {}),
- chuỗi so sau khi bỏ dấu + bỏ dấu câu (người dùng gõ không dấu thì model chép không dấu cũng đúng),
- câu tự do (câu hỏi luật, nội dung nhắc) so bằng token F1 >= 0.5,
- địa điểm / bài hát: từ của gold nằm trọn trong dự đoán ("nhạc Đen Vâu" ~ "Đen Vâu") hoặc ngược lại nhưng
  không ngắn quá nửa ("Nội Bài" ~ "sân bay Nội Bài"). Không dùng F1 vì "Hồ Tây" vs "Hồ Gươm" đã được 0.5.
"""

from .parse import Action, ToolCall
from .text import alnum, normalize, token_f1
from .tools import FREE_TEXT_ARGS, NAME_ARGS, canonical_args

FREE_TEXT_MIN_F1 = 0.5


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _name_equal(gold: str, pred: str) -> bool:
    g, p = set(normalize(gold).split()), set(normalize(pred).split())
    if not g or not p:
        return False
    return g <= p or (p <= g and len(p) * 2 >= len(g))


def arg_equal(tool: str, key: str, gold, pred) -> bool:
    if (tool, key) in FREE_TEXT_ARGS:
        return isinstance(pred, str) and token_f1(pred, str(gold)) >= FREE_TEXT_MIN_F1
    if (tool, key) in NAME_ARGS:
        return isinstance(pred, str) and _name_equal(str(gold), pred)
    if isinstance(gold, bool) or isinstance(pred, bool):
        return gold is pred
    g, p = _num(gold), _num(pred)
    if isinstance(gold, int | float):
        # "22" (chuỗi) vẫn tính đúng giá trị; lỗi sai kiểu được đếm riêng ở metric schema
        return p is not None and abs(g - p) < 1e-6
    if isinstance(gold, str) and isinstance(pred, str):
        return normalize(gold) == normalize(pred) or (alnum(gold) != "" and alnum(gold) == alnum(pred))
    return gold == pred


def call_equal(gold: ToolCall, pred: ToolCall) -> bool:
    if gold.name != pred.name or not isinstance(pred.arguments, dict):
        return False
    ga, pa = canonical_args(gold.name, gold.arguments), canonical_args(pred.name, pred.arguments)
    if set(ga) != set(pa):
        return False
    return all(arg_equal(gold.name, k, ga[k], pa[k]) for k in ga)


def calls_equal(gold: list[ToolCall], pred: list[ToolCall]) -> bool:
    if len(gold) != len(pred):
        return False
    rest = list(pred)
    for g in gold:
        hit = next((p for p in rest if call_equal(g, p)), None)
        if hit is None:
            return False
        rest.remove(hit)
    return True


def actions_match(gold: Action, pred: Action) -> bool:
    if gold.kind != pred.kind:
        return False
    return gold.kind != "call" or calls_equal(gold.calls, pred.calls)
