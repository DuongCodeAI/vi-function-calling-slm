"""Trạng thái xe tại thời điểm người dùng nói. Model cần biết để quyết định gọi hay từ chối.

State được render vào lượt user (thẻ <xe>...</xe>) chứ không vào system prompt:
system + schema tool là phần dài nhất (~1.5k token) và không đổi, giữ nguyên thì
llama.cpp tái dùng KV cache của prefix, mỗi lượt chỉ phải prefill vài chục token.
"""

from dataclasses import asdict, dataclass, fields


@dataclass
class VehicleState:
    speed_kmh: float = 0
    gear: str = "P"  # P/R/N/D
    is_night: bool = False
    battery_pct: int = 80
    range_km: int | None = None
    doors_locked: bool = True
    cabin_temp_c: float | None = None
    outside_temp_c: float | None = None

    @property
    def moving(self) -> bool:
        return self.speed_kmh > 0

    def render(self) -> str:
        parts = [f"tốc độ {self.speed_kmh:g} km/h", f"số {self.gear}", "ban đêm" if self.is_night else "ban ngày"]
        pin = f"pin {self.battery_pct}%"
        if self.range_km is not None:
            pin += f" (còn ~{self.range_km} km)"
        parts.append(pin)
        parts.append("cửa đã khoá" if self.doors_locked else "cửa chưa khoá")
        if self.cabin_temp_c is not None:
            parts.append(f"trong xe {self.cabin_temp_c:g}°C")
        if self.outside_temp_c is not None:
            parts.append(f"ngoài trời {self.outside_temp_c:g}°C")
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "VehicleState":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def as_state(state) -> VehicleState:
    if isinstance(state, VehicleState):
        return state
    return VehicleState.from_dict(state)
