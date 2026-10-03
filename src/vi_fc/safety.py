"""Luật an toàn theo trạng thái xe.

Dùng ở 2 chỗ:
- sinh nhãn gold: request + state vi phạm luật -> nhãn "refuse" (hoặc "ask" để xác nhận),
- guard lúc chạy: model vẫn gọi tool dù luật bảo từ chối -> ghi đè thành từ chối.
Model học từ chối để câu trả lời tự nhiên, còn luật là lớp chặn cuối, không tin hoàn toàn vào model 1.7B.

Đây là vài luật minh hoạ do mình đặt ra, không phải đặc tả an toàn của hãng xe nào.
"""

from dataclasses import dataclass

from .state import VehicleState

WINDOW_CONFIRM_SPEED = 80  # km/h, trên mức này mở cửa sổ quá nửa thì hỏi lại
WINDOW_CONFIRM_PERCENT = 50


@dataclass
class Verdict:
    decision: str  # allow | refuse | confirm
    rule: str = ""
    message: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


ALLOW = Verdict("allow")


def check_call(name: str, args: dict | None, state: VehicleState) -> Verdict:
    args = args or {}
    speed = f"{state.speed_kmh:g}"
    if name == "unlock_doors" and state.moving:
        return Verdict("refuse", "unlock_while_moving",
                       f"Xe đang chạy {speed} km/h, mở khoá cửa lúc này không an toàn. "
                       "Bạn dừng hẳn xe rồi mình mở nhé.")
    if name == "set_lights" and args.get("mode") == "off" and state.is_night and state.moving:
        return Verdict("refuse", "lights_off_at_night",
                       "Trời tối mà xe đang chạy, tắt đèn lúc này không an toàn nên mình giữ đèn bật.")
    if name == "open_window" and state.speed_kmh > WINDOW_CONFIRM_SPEED:
        pct = args.get("percent", 100)
        if isinstance(pct, int | float) and pct >= WINDOW_CONFIRM_PERCENT:
            # không cấm hẳn, chỉ hỏi lại: gió và tiếng ồn lớn nhưng không nguy hiểm trực tiếp
            return Verdict("confirm", "window_high_speed",
                           f"Xe đang chạy {speed} km/h, mở cửa sổ {pct:g}% sẽ rất ồn và gió mạnh. "
                           "Bạn chắc chắn muốn mở chứ?")
    return ALLOW


def check_calls(calls: list, state: VehicleState) -> Verdict:
    """Nhiều lệnh trong một câu: lấy kết quả nghiêm nhất (refuse > confirm > allow)."""
    worst = ALLOW
    for name, args in calls:
        v = check_call(name, args, state)
        if v.decision == "refuse":
            return v
        if v.decision == "confirm" and worst.allowed:
            worst = v
    return worst
