from vi_fc.safety import check_call, check_calls
from vi_fc.state import VehicleState


def test_unlock_only_when_stopped():
    assert check_call("unlock_doors", {}, VehicleState(speed_kmh=0)).allowed
    v = check_call("unlock_doors", {}, VehicleState(speed_kmh=40, gear="D"))
    assert v.decision == "refuse" and "40 km/h" in v.message


def test_lights_off_at_night_moving():
    assert check_call("set_lights", {"mode": "off"}, VehicleState(speed_kmh=50, is_night=True)).decision == "refuse"
    assert check_call("set_lights", {"mode": "off"}, VehicleState(speed_kmh=50, is_night=False)).allowed
    assert check_call("set_lights", {"mode": "off"}, VehicleState(speed_kmh=0, is_night=True)).allowed
    assert check_call("set_lights", {"mode": "low"}, VehicleState(speed_kmh=50, is_night=True)).allowed


def test_window_high_speed_needs_confirm():
    fast = VehicleState(speed_kmh=100, gear="D")
    assert check_call("open_window", {"position": "driver"}, fast).decision == "confirm"  # mặc định 100%
    assert check_call("open_window", {"position": "driver", "percent": 20}, fast).allowed
    assert check_call("open_window", {"position": "driver", "percent": 0}, fast).allowed  # đóng thì luôn được
    assert check_call("open_window", {"position": "all"}, VehicleState(speed_kmh=60)).allowed


def test_multi_call_takes_worst():
    st = VehicleState(speed_kmh=100, is_night=True)
    calls = [("open_window", {"position": "driver"}), ("unlock_doors", {}), ("set_volume", {"level": 5})]
    assert check_calls(calls, st).decision == "refuse"
    assert check_calls(calls[:1], st).decision == "confirm"
    assert check_calls([("set_volume", {"level": 5})], st).allowed


def test_state_render_roundtrip():
    st = VehicleState(speed_kmh=60, gear="D", is_night=True, battery_pct=45, range_km=180, cabin_temp_c=28)
    r = st.render()
    assert "60 km/h" in r and "ban đêm" in r and "còn ~180 km" in r and "28°C" in r
    assert VehicleState.from_dict({**st.to_dict(), "unknown": 1}) == st
