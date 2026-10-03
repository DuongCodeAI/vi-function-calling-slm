"""Danh sách tool trong xe (JSON schema kiểu OpenAI) + validator tự viết.

Mô tả để ngắn: toàn bộ schema được nhét vào prompt mỗi lượt, model 1.7B chạy CPU thì
mỗi token prefill đều tốn thời gian (xem NOTES, phần độ trễ).
Không dùng thư viện jsonschema: chỉ cần type/enum/min/max/required, tự viết ~50 dòng
và báo lỗi bằng tiếng Việt dễ đọc hơn.
"""

import copy

_INT = "integer"
_STR = "string"
_BOOL = "boolean"


def _fn(name: str, desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []},
    }


TOOLS: list[dict] = [
    _fn("set_climate", "Bật/tắt và chỉnh điều hoà.", {
        "power": {"type": _STR, "enum": ["on", "off"], "description": "mặc định on"},
        "temperature": {"type": "number", "minimum": 16, "maximum": 30, "description": "độ C"},
        "zone": {"type": _STR, "enum": ["driver", "passenger", "all"]},
        "mode": {"type": _STR, "enum": ["auto", "cool", "heat"]},
    }),
    _fn("set_fan_speed", "Chỉnh mức gió điều hoà.", {
        "level": {"type": _INT, "minimum": 1, "maximum": 7},
    }, ["level"]),
    _fn("navigate_to", "Dẫn đường tới một địa điểm.", {
        "destination": {"type": _STR, "minLength": 1},
        "avoid_tolls": {"type": _BOOL, "description": "tránh trạm thu phí"},
    }, ["destination"]),
    _fn("find_charging_station", "Tìm trạm sạc gần nhất.", {
        "max_distance_km": {"type": _INT, "minimum": 1, "maximum": 100},
        "fast_only": {"type": _BOOL, "description": "chỉ trạm sạc nhanh"},
    }),
    _fn("play_music", "Phát nhạc/podcast. Không có query thì phát tiếp.", {
        "query": {"type": _STR, "minLength": 1, "description": "tên bài, ca sĩ, thể loại"},
    }),
    _fn("set_volume", "Chỉnh âm lượng loa.", {
        "level": {"type": _INT, "minimum": 0, "maximum": 30},
    }, ["level"]),
    _fn("call_contact", "Gọi điện cho người trong danh bạ.", {
        "name": {"type": _STR, "minLength": 1},
    }, ["name"]),
    _fn("read_messages", "Đọc tin nhắn mới.", {
        "count": {"type": _INT, "minimum": 1, "maximum": 10},
    }),
    _fn("get_vehicle_status", "Xem thông tin xe.", {
        "field": {"type": _STR, "enum": ["battery", "range", "tire_pressure", "odometer"]},
    }, ["field"]),
    _fn("open_window", "Mở/đóng cửa sổ. percent=0 là đóng hẳn, mặc định 100.", {
        "position": {"type": _STR, "enum": ["driver", "passenger", "rear_left", "rear_right", "all"]},
        "percent": {"type": _INT, "minimum": 0, "maximum": 100},
    }, ["position"]),
    _fn("lock_doors", "Khoá tất cả cửa."),
    _fn("unlock_doors", "Mở khoá tất cả cửa."),
    _fn("set_lights", "Chỉnh đèn pha.", {
        "mode": {"type": _STR, "enum": ["off", "auto", "low", "high"], "description": "low=cốt, high=pha"},
    }, ["mode"]),
    _fn("set_drive_mode", "Đổi chế độ lái.", {
        "mode": {"type": _STR, "enum": ["eco", "comfort", "sport"]},
    }, ["mode"]),
    _fn("set_reminder", "Hẹn giờ nhắc việc.", {
        "text": {"type": _STR, "minLength": 1},
        "minutes": {"type": _INT, "minimum": 1, "maximum": 1440, "description": "nhắc sau bao nhiêu phút"},
    }, ["text", "minutes"]),
    _fn("lookup_traffic_law", "Tra luật giao thông Việt Nam (mức phạt, quy định).", {
        "question": {"type": _STR, "minLength": 1},
    }, ["question"]),
    _fn("lookup_sign", "Tra ý nghĩa biển báo theo số hiệu, vd P.102, W.207a.", {
        "code": {"type": _STR, "minLength": 1},
    }, ["code"]),
]

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}
TOOL_NAMES = list(TOOLS_BY_NAME)

# tham số là câu tự do: khi chấm điểm so bằng token F1 thay vì so khớp tuyệt đối
FREE_TEXT_ARGS = {("lookup_traffic_law", "question"), ("set_reminder", "text")}
# tên địa điểm / bài hát: "Đen Vâu" và "nhạc Đen Vâu" đều tìm ra đúng thứ cần -> chấm theo tập con từ
NAME_ARGS = {("navigate_to", "destination"), ("play_music", "query")}

# giá trị mặc định: "bật điều hoà" ra {} hay {"power": "on"} đều đúng -> bỏ trước khi so
DEFAULTS = {("set_climate", "power"): "on", ("open_window", "percent"): 100}


def canonical_args(name: str, args: dict) -> dict:
    return {k: v for k, v in args.items() if DEFAULTS.get((name, k), object()) != v}


def openai_tools(names: list[str] | None = None) -> list[dict]:
    """Dạng `tools=` cho apply_chat_template của Qwen3 và API kiểu OpenAI."""
    picked = TOOLS if names is None else [TOOLS_BY_NAME[n] for n in names]
    return [{"type": "function", "function": copy.deepcopy(t)} for t in picked]


def _type_ok(value, typ: str) -> bool:
    # bool là con của int trong Python -> phải loại riêng, không thì True lọt qua "integer"
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if typ == "string":
        return isinstance(value, str)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "object":
        return isinstance(value, dict)
    if typ == "array":
        return isinstance(value, list)
    raise ValueError(f"type chưa hỗ trợ: {typ}")


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Trả về danh sách lỗi (rỗng = hợp lệ). Object không cho phép key lạ:
    model hay bịa thêm tham số (vd `unit: "celsius"`), cần bắt được."""
    errors: list[str] = []
    typ = schema.get("type")
    if typ and not _type_ok(value, typ):
        return [f"{path}: cần kiểu {typ}, nhận {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} không thuộc {schema['enum']}")
    if typ in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > {schema['maximum']}")
    if typ == "string" and len(value.strip()) < schema.get("minLength", 0):
        errors.append(f"{path}: chuỗi rỗng")
    if typ == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: thiếu '{key}'")
        for key, v in value.items():
            if key not in props:
                errors.append(f"{path}: tham số lạ '{key}'")
            else:
                errors += validate(v, props[key], f"{path}.{key}")
    if typ == "array" and "items" in schema:
        for i, v in enumerate(value):
            errors += validate(v, schema["items"], f"{path}[{i}]")
    return errors


def validate_call(name: str, args) -> list[str]:
    if name not in TOOLS_BY_NAME:
        return [f"tool không tồn tại: {name}"]
    return validate(args, TOOLS_BY_NAME[name]["parameters"], name)
