"""Kho giá trị slot + cách nói tiếng Việt của từng giá trị enum.

ALIASES dùng 2 việc: phần tử đầu là cách diễn đạt đưa cho LLM trong đề bài, cả danh sách dùng để
kiểm tra câu LLM viết ra có thật sự nhắc tới giá trị đó không (lọc nhãn sai).
"""

DESTINATIONS = [
    "Hồ Gươm", "sân bay Nội Bài", "sân bay Tân Sơn Nhất", "chợ Bến Thành", "Landmark 81", "bệnh viện Bạch Mai",
    "Times City", "Vincom Bà Triệu", "phố cổ Hội An", "cầu Rồng", "Đại học Bách khoa Hà Nội", "nhà thờ Đức Bà",
    "Hồ Tây", "bến xe Mỹ Đình", "Aeon Mall Long Biên", "công viên Thống Nhất", "chợ Đồng Xuân", "Ocean Park",
    "Phú Mỹ Hưng", "Thảo Cầm Viên", "Royal City", "biển Mỹ Khê", "đèo Hải Vân", "số 5 Lê Duẩn", "nhà", "công ty",
    "trường Tiểu học Kim Đồng", "bệnh viện Chợ Rẫy", "Bà Nà Hills", "Nhà hát Lớn", "sân Mỹ Đình", "Hồ Con Rùa",
]
CONTACTS = [
    "mẹ", "bố", "vợ", "chồng", "anh Tuấn", "chị Lan", "sếp Hùng", "Minh", "bà nội", "cô Hạnh", "anh Khoa",
    "Trang", "em Ngọc", "chú Ba", "Quang Huy", "chị Thảo", "thầy Nam", "Phương Anh",
]
MUSIC = [
    "Sơn Tùng", "nhạc Trịnh", "Lạc Trôi", "nhạc không lời", "bolero", "Đen Vâu", "nhạc lofi", "Hà Anh Tuấn",
    "nhạc thiếu nhi", "podcast tin tức", "Mỹ Tâm", "Chúng Ta Của Hiện Tại", "nhạc EDM", "nhạc xưa",
    "Hoàng Thùy Linh", "nhạc Hàn", "Taylor Swift", "nhạc rap Việt",
]
REMINDERS = [
    "mua sữa cho con", "gọi lại cho khách", "sạc xe", "đón con ở trường", "uống thuốc", "họp với team",
    "lấy đồ giặt", "đóng tiền điện", "nghỉ chân cho đỡ mỏi", "mua hoa cho vợ", "gửi báo cáo cho sếp",
    "ghé tiệm thuốc", "rút tiền", "đặt bánh sinh nhật",
]
MINUTES = [5, 10, 15, 20, 30, 45, 60, 90, 120]

LAW_TOPICS = [
    "xe máy vượt đèn đỏ phạt bao nhiêu", "ô tô không thắt dây an toàn bị phạt thế nào",
    "uống bia rồi lái ô tô bị phạt gì", "đi vào làn khẩn cấp trên cao tốc", "quên mang bằng lái",
    "dùng điện thoại khi đang lái xe", "chạy quá tốc độ 15 km/h", "đỗ ô tô trên vỉa hè", "đi ngược chiều",
    "quay đầu xe ở nơi có biển cấm quay đầu", "trẻ em dưới 10 tuổi ngồi ghế trước ô tô",
    "không nhường đường xe cứu thương", "bấm còi trong khu dân cư ban đêm", "bị trừ điểm bằng lái",
    "tốc độ tối đa trong khu dân cư", "dừng xe trên cao tốc", "lùi xe trên cao tốc", "vượt bên phải",
    "chuyển làn không bật xi nhan", "đi sai làn đường", "không có bảo hiểm xe", "chở quá số người",
]
SIGN_CODES = [
    "P.102", "P.103a", "P.106a", "P.123a", "P.124a", "P.127", "P.130", "P.131a", "R.302a", "R.303",
    "R.411", "R.434", "W.201a", "W.207b", "W.224", "W.245a", "I.408", "I.423b",
]
CHITCHAT_TOPICS = [
    "chào hỏi trợ lý buổi sáng", "hỏi trợ lý tên gì", "nhờ kể một câu chuyện cười", "khen trợ lý giỏi",
    "hỏi thời tiết ngày mai", "nhờ đặt vé máy bay", "hỏi giá vàng hôm nay", "than mệt vì kẹt xe",
    "hỏi cách nấu phở", "cảm ơn trợ lý", "hỏi đội bóng nào vô địch năm nay", "nhờ đặt bàn nhà hàng",
    "hỏi trợ lý có biết hát không", "kể là hôm nay sinh nhật mình", "hỏi nên ăn trưa món gì",
]

ALIASES: dict[tuple[str, str], dict] = {
    ("set_climate", "power"): {"on": ["bật", "mở", "on"], "off": ["tắt", "off", "ngắt"]},
    ("set_climate", "zone"): {
        "driver": ["ghế lái", "bên lái", "tài xế", "tài", "driver"],
        "passenger": ["ghế phụ", "bên phụ", "phụ", "passenger", "bên cạnh"],
        "all": ["cả xe", "tất cả", "toàn bộ", "cả hai", "all", "hết", "mọi"],
    },
    ("set_climate", "mode"): {
        "auto": ["tự động", "auto"], "cool": ["làm mát", "mát", "lạnh", "cool"],
        "heat": ["sưởi", "ấm", "nóng", "heat"],
    },
    ("get_vehicle_status", "field"): {
        "battery": ["pin", "battery", "phần trăm"],
        "range": ["quãng đường còn đi được", "đi được", "quãng đường", "range", "bao xa", "bao nhiêu km",
                  "bao nhiêu cây", "chạy được"],
        "tire_pressure": ["áp suất lốp", "lốp", "bánh", "tire", "vỏ xe"],
        "odometer": ["số km đã chạy", "odo", "đã chạy", "công tơ", "tổng"],
    },
    ("open_window", "position"): {
        "driver": ["cửa sổ ghế lái", "ghế lái", "bên lái", "tài", "driver", "lái"],
        "passenger": ["cửa sổ ghế phụ", "ghế phụ", "bên phụ", "phụ", "passenger"],
        "rear_left": ["cửa sổ sau bên trái", "sau bên trái", "sau trái", "trái phía sau", "phía sau bên trái",
                      "rear left", "sau ben trai"],
        "rear_right": ["cửa sổ sau bên phải", "sau bên phải", "sau phải", "phải phía sau", "phía sau bên phải",
                       "rear right"],
        "all": ["tất cả cửa sổ", "tất cả", "hết", "cả", "toàn bộ", "all", "4 cửa", "bốn cửa", "các cửa"],
    },
    ("set_lights", "mode"): {
        "off": ["tắt đèn", "tắt", "off"], "auto": ["tự động", "auto"],
        "low": ["đèn cốt", "cốt", "chiếu gần", "gần", "low"], "high": ["đèn pha", "pha", "chiếu xa", "xa", "high"],
    },
    ("set_drive_mode", "mode"): {
        "eco": ["eco", "tiết kiệm"], "comfort": ["comfort", "thoải mái", "tiêu chuẩn", "bình thường", "normal"],
        "sport": ["sport", "thể thao"],
    },
    ("navigate_to", "avoid_tolls"): {True: ["tránh trạm thu phí", "phí", "bot", "toll", "trạm"]},
    ("find_charging_station", "fast_only"): {True: ["chỉ trạm sạc nhanh", "nhanh", "fast", "dc"]},
}

# số có cách nói riêng ngoài chữ số/chữ đọc
NUMBER_ALIASES: dict[tuple[str, str], dict[int, list[str]]] = {
    ("open_window", "percent"): {0: ["đóng", "kéo lên", "close"], 50: ["một nửa", "nửa"],
                                 100: ["hết", "toàn bộ", "hoàn toàn"]},
    ("set_reminder", "minutes"): {30: ["nửa tiếng"], 60: ["một tiếng", "1 tiếng", "1 giờ", "một giờ"],
                                  90: ["tiếng rưỡi"], 120: ["hai tiếng", "2 tiếng", "2 giờ", "hai giờ"]},
}

# cách mô tả tool/tham số trong đề bài gửi LLM: (nhãn, đơn vị)
TOOL_PHRASE = {
    "set_climate": "chỉnh điều hoà", "set_fan_speed": "chỉnh mức gió điều hoà", "navigate_to": "dẫn đường",
    "find_charging_station": "tìm trạm sạc", "play_music": "mở nhạc", "set_volume": "chỉnh âm lượng",
    "call_contact": "gọi điện", "read_messages": "đọc tin nhắn mới", "get_vehicle_status": "hỏi thông tin xe",
    "open_window": "mở cửa sổ", "lock_doors": "khoá cửa xe", "unlock_doors": "mở khoá cửa xe",
    "set_lights": "chỉnh đèn xe", "set_drive_mode": "đổi chế độ lái", "set_reminder": "hẹn nhắc việc",
    "lookup_traffic_law": "hỏi luật giao thông", "lookup_sign": "hỏi ý nghĩa biển báo",
}
ARG_LABEL = {
    ("set_climate", "temperature"): ("nhiệt độ", " độ"), ("set_fan_speed", "level"): ("mức gió", ""),
    ("navigate_to", "destination"): ("điểm đến", ""),
    ("find_charging_station", "max_distance_km"): ("trong vòng", " km"),
    ("play_music", "query"): ("bài/ca sĩ/thể loại", ""), ("set_volume", "level"): ("mức âm lượng", ""),
    ("call_contact", "name"): ("người cần gọi", ""), ("read_messages", "count"): ("số tin nhắn", ""),
    ("get_vehicle_status", "field"): ("thông tin cần xem", ""), ("open_window", "position"): ("cửa sổ nào", ""),
    ("open_window", "percent"): ("mức mở", "%"), ("set_lights", "mode"): ("chế độ đèn", ""),
    ("set_drive_mode", "mode"): ("chế độ lái", ""), ("set_reminder", "text"): ("nội dung cần nhắc", ""),
    ("set_reminder", "minutes"): ("sau bao lâu", " phút"), ("lookup_sign", "code"): ("số hiệu biển", ""),
}
