"""Xử lý chuỗi tiếng Việt dùng chung: bỏ dấu, chuẩn hoá, n-gram, đọc số thành chữ."""

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def fold(s: str) -> str:
    """Bỏ dấu + lowercase. 'đ' không phải dấu kết hợp nên NFD không tách được, phải thay tay."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


def normalize(s: str, keep_diacritics: bool = False) -> str:
    s = unicodedata.normalize("NFC", s.lower()) if keep_diacritics else fold(s)
    s = _PUNCT.sub(" ", s).replace("_", " ")
    return _SPACES.sub(" ", s).strip()


def alnum(s: str) -> str:
    """'P.131a' / 'p 131 a' -> 'p131a' (so mã biển báo)."""
    return re.sub(r"[^a-z0-9]", "", fold(s))


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = f" {normalize(s)} "
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = 0
    rest = list(g)
    for t in p:
        if t in rest:
            rest.remove(t)
            common += 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


_UNITS = ["khong", "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin"]


def number_words(n: int) -> set[str]:
    """Các cách đọc số 0-100 (đã bỏ dấu). Chỉ dùng để kiểm tra câu do LLM viết có nhắc đúng giá trị không,
    nên chấp nhận cả cách nói tắt kiểu "hai hai" (22), "hai lam" (25)."""
    if n < 0 or n > 100:
        return set()
    if n < 10:
        return {_UNITS[n]}
    if n == 100:
        return {"mot tram", "tram"}
    tens, unit = divmod(n, 10)
    if tens == 1:
        head = {"muoi"}
        tails = {"lam", "nham"} if unit == 5 else {_UNITS[unit]}
        return head if unit == 0 else {f"muoi {t}" for t in tails}
    t = _UNITS[tens]
    if unit == 0:
        return {f"{t} muoi", f"{t} chuc"}
    # 21 = "hai mốt" (bỏ dấu thành "mot", trùng chữ "một"), 24 = "hai tư", 25 = "hai lăm"
    tails = {"tu", "bon"} if unit == 4 else {"lam", "nham"} if unit == 5 else {_UNITS[unit]}
    out = set()
    for u in tails:
        out |= {f"{t} muoi {u}", f"{t} {u}"}
    return out


def mentions_number(text: str, n: int | float) -> bool:
    """Câu có nhắc tới số n không (chữ số hoặc chữ). Kiểm tra trên text đã bỏ dấu."""
    t = normalize(text)
    if float(n).is_integer():
        n = int(n)
        if re.search(rf"(?<!\d){n}(?!\d)", t):
            return True
        return any(re.search(rf"\b{w}\b", t) for w in number_words(n))
    # số lẻ kiểu 22.5 -> "22,5" / "22.5" / "22 rưỡi"
    whole = int(n)
    raw = fold(text)
    return bool(re.search(rf"{whole}[.,]5|{whole} ruoi", raw))
