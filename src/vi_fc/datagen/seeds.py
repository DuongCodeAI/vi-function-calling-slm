"""Seed = (kịch bản, tool, giá trị slot, trạng thái xe, văn phong) do code bốc ngẫu nhiên.

Nhãn gold dựng từ seed, KHÔNG để LLM tự gán: LLM chỉ viết câu người dùng. Như vậy nhãn luôn
khớp schema, khớp luật an toàn, và phân phối tool/kịch bản do mình kiểm soát. Rủi ro còn lại là
LLM viết câu không khớp seed (quên giá trị, tự thêm yêu cầu) -> lọc ở filters.py.
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

from ..parse import Action, ToolCall
from ..safety import check_call, check_calls
from ..state import VehicleState
from ..tools import TOOLS_BY_NAME
from . import values as V

SCENARIOS = {
    "single": 0.34,  # 1 lệnh đủ thông tin
    "multi": 0.12,  # 2 lệnh trong 1 câu
    "missing": 0.12,  # thiếu slot bắt buộc -> hỏi lại
    "followup": 0.08,  # lượt trước đã hỏi lại, lượt này người dùng bổ sung -> gọi
    "unsafe": 0.12,  # vi phạm luật an toàn theo state -> từ chối / xác nhận
    "law": 0.11,  # hỏi luật -> lookup_traffic_law
    "chitchat": 0.11,  # ngoài phạm vi -> trả lời thường
}
STYLES = {
    "standard": "câu bình thường, rõ ràng",
    "colloquial": "văn nói đời thường, có từ đệm (ơi, nhé, đi, xíu, cái), có thể nói trống không",
    "code_switch": "chen từ tiếng Anh kiểu dân văn phòng (AC, volume, navigation, mode, fan, call, play)",
    "southern": "giọng miền Nam (nha, nghen, hen, máy lạnh, quẹo, mở nhạc, kêu)",
    "northern": "giọng miền Bắc (nhé, điều hoà, rẽ, bật, gọi cho)",
    "short": "ra lệnh rất ngắn, 3-7 chữ, vẫn đủ giá trị",
}
STYLE_WEIGHTS = {"standard": 0.2, "colloquial": 0.25, "code_switch": 0.15, "southern": 0.15, "northern": 0.15,
                 "short": 0.1}

# slot bắt buộc mà thiếu thì nên hỏi lại (không phải slot nào thiếu cũng hỏi: "mở nhạc" thì cứ phát)
MISSING_SLOTS = [
    ("set_fan_speed", "level"), ("navigate_to", "destination"), ("set_volume", "level"), ("call_contact", "name"),
    ("get_vehicle_status", "field"), ("open_window", "position"), ("set_lights", "mode"),
    ("set_drive_mode", "mode"), ("set_reminder", "text"), ("set_reminder", "minutes"),
]
ASK_TEMPLATES = {
    ("set_fan_speed", "level"): "Bạn muốn để quạt mức mấy, từ 1 đến 7?",
    ("navigate_to", "destination"): "Bạn muốn đi đến đâu?",
    ("set_volume", "level"): "Bạn muốn chỉnh âm lượng mức bao nhiêu, từ 0 đến 30?",
    ("call_contact", "name"): "Bạn muốn gọi cho ai?",
    ("get_vehicle_status", "field"): "Bạn muốn xem gì: pin, quãng đường còn đi được, áp suất lốp hay số km đã chạy?",
    ("open_window", "position"): "Bạn muốn mở cửa sổ nào?",
    ("set_lights", "mode"): "Bạn muốn chỉnh đèn sang chế độ nào: tắt, tự động, cốt hay pha?",
    ("set_drive_mode", "mode"): "Bạn muốn chuyển sang chế độ lái nào: eco, comfort hay sport?",
    ("set_reminder", "text"): "Bạn muốn mình nhắc việc gì?",
    ("set_reminder", "minutes"): "Bạn muốn mình nhắc sau bao lâu?",
}
SINGLE_TOOLS = [n for n in TOOLS_BY_NAME if n != "lookup_traffic_law"]
# cặp hay đi cùng nhau trong một câu; bỏ lookup_* và cặp mâu thuẫn (khoá + mở khoá)
MULTI_TOOLS = ["set_climate", "set_fan_speed", "navigate_to", "find_charging_station", "play_music", "set_volume",
               "call_contact", "read_messages", "open_window", "lock_doors", "set_lights", "set_drive_mode",
               "set_reminder", "get_vehicle_status"]


@dataclass
class Seed:
    id: str
    scenario: str
    calls: list[dict]  # lệnh người dùng muốn: [{"name", "arguments"}]
    state: dict
    style: str
    omitted: str | None = None  # slot bị bỏ (missing / followup)
    topic: str | None = None  # law / chitchat
    extra: dict = field(default_factory=dict)

    @property
    def group(self) -> str:
        """Khoá để chia train/val/test: cùng nội dung yêu cầu (bỏ qua văn phong, state) thì cùng một phía."""
        calls = ";".join(f"{c['name']}{json.dumps(c['arguments'], sort_keys=True, ensure_ascii=False)}"
                         for c in self.calls)
        return f"{self.scenario}|{calls}|{self.omitted or ''}|{self.topic or ''}"

    def to_dict(self) -> dict:
        return asdict(self)


def sample_args(name: str, rng: random.Random) -> dict:
    a: dict = {}
    if name == "set_climate":
        if rng.random() < 0.15:
            return {"power": "off"}
        if rng.random() < 0.75:
            a["temperature"] = rng.choice(range(16, 31)) if rng.random() < 0.9 else rng.choice([20.5, 22.5, 24.5])
        if rng.random() < 0.3:
            a["zone"] = rng.choice(["driver", "passenger", "all"])
        if rng.random() < 0.3:
            a["mode"] = rng.choice(["auto", "cool", "heat"])
    elif name == "set_fan_speed":
        a["level"] = rng.randint(1, 7)
    elif name == "navigate_to":
        a["destination"] = rng.choice(V.DESTINATIONS)
        if rng.random() < 0.2:
            a["avoid_tolls"] = True
    elif name == "find_charging_station":
        if rng.random() < 0.4:
            a["max_distance_km"] = rng.choice([5, 10, 20, 30, 50])
        if rng.random() < 0.4:
            a["fast_only"] = True
    elif name == "play_music":
        if rng.random() < 0.85:
            a["query"] = rng.choice(V.MUSIC)
    elif name == "set_volume":
        a["level"] = rng.randint(0, 30)
    elif name == "call_contact":
        a["name"] = rng.choice(V.CONTACTS)
    elif name == "read_messages":
        if rng.random() < 0.4:
            a["count"] = rng.choice([1, 2, 3, 5])
    elif name == "get_vehicle_status":
        a["field"] = rng.choice(["battery", "range", "tire_pressure", "odometer"])
    elif name == "open_window":
        a["position"] = rng.choice(["driver", "passenger", "rear_left", "rear_right", "all"])
        if rng.random() < 0.6:
            a["percent"] = rng.choice([0, 20, 30, 50, 100])
    elif name in ("set_lights", "set_drive_mode"):
        a["mode"] = rng.choice(TOOLS_BY_NAME[name]["parameters"]["properties"]["mode"]["enum"])
    elif name == "set_reminder":
        a["text"] = rng.choice(V.REMINDERS)
        a["minutes"] = rng.choice(V.MINUTES)
    elif name == "lookup_sign":
        a["code"] = rng.choice(V.SIGN_CODES)
    return a


def sample_state(rng: random.Random, **override) -> dict:
    speed = override.pop("speed_kmh", None)
    if speed is None:
        speed = rng.choice([0, 0, 0, 20, 40, 60, 80, 100, 120])
    battery = rng.randint(5, 100)
    st = VehicleState(
        speed_kmh=speed,
        gear="D" if speed > 0 else rng.choice(["P", "P", "D"]),
        is_night=rng.random() < 0.35,
        battery_pct=battery,
        range_km=int(battery * 4.2),
        doors_locked=True if speed > 0 else rng.random() < 0.5,
        cabin_temp_c=rng.randint(18, 36),
        outside_temp_c=rng.randint(15, 39),
    )
    d = st.to_dict()
    d.update(override)
    return d


def _safe_state(calls: list[dict], rng: random.Random, tries: int = 30) -> dict:
    """State mà mọi lệnh đều được phép. Tool như unlock_doors chỉ hợp lệ khi xe đứng yên ->
    tự nhiên sinh ra cặp tương phản với kịch bản unsafe (cùng câu, khác state, khác nhãn)."""
    for _ in range(tries):
        st = sample_state(rng)
        if check_calls([(c["name"], c["arguments"]) for c in calls], VehicleState.from_dict(st)).allowed:
            return st
    return sample_state(rng, speed_kmh=0, is_night=False)


def _pick(rng: random.Random, weights: dict) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _unsafe(rng: random.Random) -> tuple[list[dict], dict]:
    kind = rng.choice(["unlock", "unlock", "lights", "window"])
    if kind == "unlock":
        return [{"name": "unlock_doors", "arguments": {}}], sample_state(rng, speed_kmh=rng.choice([10, 30, 50, 80]))
    if kind == "lights":
        st = sample_state(rng, speed_kmh=rng.choice([20, 40, 60, 90]), is_night=True)
        return [{"name": "set_lights", "arguments": {"mode": "off"}}], st
    args = {"position": rng.choice(["driver", "passenger", "rear_left", "rear_right", "all"])}
    pct = rng.choice([None, 50, 80, 100])
    if pct is not None:
        args["percent"] = pct
    return [{"name": "open_window", "arguments": args}], sample_state(rng, speed_kmh=rng.choice([90, 100, 110, 120]))


def sample_seed(rng: random.Random, idx: int, scenario: str | None = None) -> Seed:
    scenario = scenario or _pick(rng, SCENARIOS)
    style = _pick(rng, STYLE_WEIGHTS)
    sid = f"s{idx:06d}"
    if scenario == "single":
        name = rng.choice(SINGLE_TOOLS)
        calls = [{"name": name, "arguments": sample_args(name, rng)}]
        return Seed(sid, scenario, calls, _safe_state(calls, rng), style)
    if scenario == "multi":
        a, b = rng.sample(MULTI_TOOLS, 2)
        calls = [{"name": n, "arguments": sample_args(n, rng)} for n in (a, b)]
        return Seed(sid, scenario, calls, _safe_state(calls, rng), style)
    if scenario in ("missing", "followup"):
        name, slot = rng.choice(MISSING_SLOTS)
        args = sample_args(name, rng)
        calls = [{"name": name, "arguments": args}]
        state = _safe_state(calls, rng)
        if scenario == "missing":
            # lệnh "dự định" khi thiếu slot: dùng cho đề bài LLM và cho nhiễu DPO "đoán bừa thay vì hỏi"
            calls = [{"name": name, "arguments": {k: v for k, v in args.items() if k != slot}}]
        return Seed(sid, scenario, calls, state, style, omitted=slot, extra={"omitted_value": args[slot]})
    if scenario == "unsafe":
        calls, state = _unsafe(rng)
        return Seed(sid, scenario, calls, state, style)
    if scenario == "law":
        return Seed(sid, scenario, [{"name": "lookup_traffic_law", "arguments": {}}], sample_state(rng), style,
                    topic=rng.choice(V.LAW_TOPICS))
    if scenario == "chitchat":
        return Seed(sid, scenario, [], sample_state(rng), style, topic=rng.choice(V.CHITCHAT_TOPICS))
    raise ValueError(f"kịch bản lạ: {scenario}")


def gold_action(seed: Seed, user: str, reply: str | None = None) -> Action:
    """Nhãn đúng cho seed. `user` chỉ cần cho lookup_traffic_law (câu hỏi chuyển nguyên cho RAG)."""
    sc = seed.scenario
    if sc == "chitchat":
        return Action("reply", text=(reply or "").strip())
    if sc == "law":
        return Action("call", [ToolCall("lookup_traffic_law", {"question": user.strip()})])
    if sc == "missing":
        return Action("ask", text=ASK_TEMPLATES[(seed.calls[0]["name"], seed.omitted)])
    calls = [ToolCall(c["name"], dict(c["arguments"])) for c in seed.calls]
    verdict = check_calls(calls, VehicleState.from_dict(seed.state))
    if verdict.decision == "refuse":
        return Action("refuse", text=verdict.message)
    if verdict.decision == "confirm":
        return Action("ask", text=verdict.message)
    return Action("call", calls)


def describe_value(name: str, arg: str, value) -> str:
    if (name, arg) in V.ALIASES:
        return V.ALIASES[(name, arg)][value][0]
    if name == "open_window" and arg == "percent":
        return "đóng hẳn" if value == 0 else f"mở {value}%"
    label, unit = V.ARG_LABEL.get((name, arg), (arg, ""))
    if isinstance(value, str):
        return f'{label} "{value}"'
    return f"{label} {value:g}{unit}"


def describe_call(call: dict) -> str:
    name, args = call["name"], call["arguments"]
    if name == "open_window" and args.get("percent") == 0:
        head = "đóng cửa sổ"
        args = {k: v for k, v in args.items() if k != "percent"}
    elif name == "set_climate" and args.get("power") == "off":
        return "tắt điều hoà"
    else:
        head = V.TOOL_PHRASE[name]
    parts = [describe_value(name, k, v) for k, v in args.items()]
    return head + (": " + ", ".join(parts) if parts else "")


def seed_hash(text: str) -> float:
    """Số ổn định trong [0, 1) theo nội dung (md5), không phụ thuộc PYTHONHASHSEED."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / 16**8


def check_seed_label(seed: Seed) -> bool:
    """Seed unsafe thật sự phải bị luật chặn (phòng khi sửa luật mà quên sửa sampler)."""
    if seed.scenario != "unsafe":
        return True
    c = seed.calls[0]
    return not check_call(c["name"], c["arguments"], VehicleState.from_dict(seed.state)).allowed
