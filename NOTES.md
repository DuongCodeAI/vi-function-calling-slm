# Ghi chú quyết định

Vì sao chọn cách này, bẫy nào đã gặp hoặc đã tính trước.

## Chọn model

- Qwen3-1.7B (Apache-2.0): có sẵn định dạng tool calling trong chat template, tiếng Việt tạm ổn, Q4_K_M ~1.1GB
  vừa RAM laptop cùng lúc với model nhận dạng giọng nói. Chưa đo so sánh: 0.6B mình sợ không đủ sức đọc 17 schema
  + state; 4B nặng gấp đôi (~2.5GB Q4), decode trên CPU chậm tương ứng, quá đắt cho một lệnh "bật điều hoà".
  Nếu 1.7B sau SFT vẫn yếu thì mới thử 4B.
- Tắt thinking (`enable_thinking=False`): với lệnh ngắn, phần suy nghĩ dài gấp mấy lần câu trả lời mà không
  giúp gì, trên CPU thì mỗi token đều là thời gian chờ.
- Giữ nguyên chat template gốc thay vì tự nghĩ format JSON riêng: model đã được dạy `<tool_call>` từ trước,
  fine-tune chỉ cần dạy "gọi cái gì, khi nào".

## Dữ liệu: ai viết, ai gán nhãn

- **Không dùng Gemini để sinh dữ liệu train.** Điều khoản Gemini API cấm dùng output để phát triển model cạnh
  tranh. Groq `openai/gpt-oss-120b` là model Apache-2.0 nên output dùng train được. Gemini chỉ xuất hiện trong
  eval (cận trên).
- **LLM chỉ viết câu, code gán nhãn.** Ban đầu định để LLM sinh cả câu lẫn tool call, nhưng như vậy nhãn sai thì
  không biết (LLM có thể gọi `set_climate` với 220 độ, hoặc quên luật an toàn). Đảo lại: code bốc seed
  (kịch bản, tool, giá trị, state), nhãn suy ra từ seed + luật; LLM chỉ được nhờ viết câu người dùng.
  Ưu điểm phụ: kiểm soát được phân phối tool/kịch bản thay vì để LLM thích gì sinh nấy.
- Lỗi còn lại chuyển thành "LLM viết lệch seed". Bộ lọc `mentions_value` kiểm tra câu có nhắc từng giá trị:
  số (cả chữ số lẫn cách đọc "hai mươi hai", "hai hai", "hai lăm"), enum qua bảng cách nói ("ghế phụ", "bên phụ",
  "passenger"), tên riêng theo âm tiết đã bỏ dấu. Bảng alias thiếu thì lọc oan câu đúng → notebook 01 in ra các
  câu bị loại để xem và bổ sung alias.
- Kịch bản thiếu slot: kiểm tra ngược — câu KHÔNG được chứa giá trị bị bỏ (LLM hay "lỡ" nói luôn con số).
- Gõ không dấu và gõ sai do code làm sau khi LLM viết, không nhờ LLM: chắc chắn đúng tỉ lệ, và không đụng vào số
  hay tên riêng (không đổi nhãn).
- Gõ không dấu thì tham số chuỗi trong nhãn cũng để không dấu ("ho guom"): không dạy model tự thêm dấu cho tên
  riêng (dễ thành bịa), phần tìm địa điểm/danh bạ phía sau tự so không dấu.
- Text của nhãn hỏi lại / từ chối lấy từ template cố định, text trả lời chuyện phiếm do LLM viết. Mọi text gold
  phải được `classify_text` xếp đúng loại, không thì bỏ.

## Trạng thái xe và luật an toàn

- Luật (`safety.py`) là nguồn sự thật duy nhất, dùng cho cả nhãn và guard lúc chạy:
  mở khoá cửa khi xe đang chạy → từ chối; tắt đèn ban đêm khi đang chạy → từ chối; mở cửa sổ ≥ 50% khi > 80 km/h
  → hỏi xác nhận (ồn, gió mạnh nhưng không nguy hiểm trực tiếp nên không cấm).
- Kịch bản single tự sinh cặp tương phản: cùng "mở khoá cửa" nhưng state đứng yên → gọi tool, đang chạy → từ chối.
  Model phải đọc state chứ không học thuộc "mở khoá = từ chối".
- Phòng thủ 2 lớp: model 1.7B không đáng tin 100%, guard chặn sau output. Hệ quả cho eval: nhãn refuse của tập
  tổng hợp cũng sinh từ chính luật đó, nên "model + guard" gần như chắc chắn đạt refuse recall 1.0 — vòng tròn.
  Số đáng xem là model **không** guard và tỉ lệ từ chối thừa.
- State nằm trong lượt user (thẻ `<xe>`), không nằm trong system prompt như ý định ban đầu. Lý do: system + schema
  17 tool (~5 nghìn ký tự) là phần dài nhất và giống hệt nhau giữa các lượt; để nguyên thì llama.cpp tái dùng KV
  cache của prefix, mỗi lượt chỉ prefill phần state + câu nói. Đặt state trong system thì mỗi lần tốc độ đổi là
  prefill lại từ đầu.
- Mô tả tool để ngắn cùng lý do trên.

## Parse output

- Chấp nhận nhiều dạng: `<tool_call>` chuẩn, nhiều thẻ liền nhau, JSON trần, có ```json fence, thẻ bị cắt cụt do
  hết max_tokens, thừa dấu `}` ở cuối (phòng trước, model nhỏ hay đóng ngoặc lệch), arguments là chuỗi JSON
  (API kiểu OpenAI).
- Hỏi lại / từ chối / trả lời phân bằng từ khoá, thứ tự ưu tiên: "không an toàn"/"nguy hiểm" → từ chối; câu hỏi
  (dấu ?, hoặc kết thúc bằng nào/bao nhiêu/mấy/không...) → hỏi lại; "xin lỗi"/"không thể" → từ chối; còn lại →
  trả lời. Biết trước là sai với câu kiểu "xin lỗi, mình không đặt vé được" (ngoài phạm vi nhưng thành từ chối).
  Không thay bằng classifier vì model fine-tune đã theo đúng quy ước của dữ liệu; chỉ model gốc và Gemini chịu thiệt.

## Chấm điểm

- Thứ tự lệnh trong câu nhiều lệnh không quan trọng.
- Tham số bằng giá trị mặc định bỏ qua: `set_climate({})` == `set_climate({"power": "on"})`.
- Câu tự do (câu hỏi luật, nội dung nhắc) so bằng token F1 ≥ 0.5.
- Địa điểm / bài hát: ban đầu cũng dùng F1 ≥ 0.5, nhưng "Hồ Tây" vs "Hồ Gươm" được đúng 0.5 → tính là khớp, sai.
  Đổi sang so tập từ: gold nằm trọn trong dự đoán ("nhạc Đen Vâu" ~ "Đen Vâu") hoặc ngược lại nhưng không ngắn
  quá nửa ("Nội Bài" ~ "sân bay Nội Bài").
- Mã biển báo so sau khi bỏ hết ký tự không phải chữ/số: "P.131a" == "p131a".

## Bỏ trùng, chia split

- Jaccard trên 3-gram ký tự của câu đã bỏ dấu, ngưỡng 0.85. Chỉ so trong cùng nhóm (loại kết quả + tên tool):
  cùng câu nhưng khác state, khác nhãn là dữ liệu tương phản có ích, không được xoá.
- Câu chỉ khác nhau ở con số ("... 22 độ" / "... 23 độ") với câu dài thì bị coi là trùng. Cố ý: LLM hay lặp
  một khuôn câu cho nhiều seed, giữ lại hết thì model học thuộc khuôn.
- Chia theo `group` = nội dung seed (kịch bản + tool + giá trị), hash md5 cố định. Câu viết từ cùng một yêu cầu
  luôn cùng phía; thêm dữ liệu mới không xáo bản ghi cũ sang split khác.

## Train

- QLoRA r=16, alpha=32, mọi lớp linear, 2 epoch, lr 2e-4. Chỉ tính loss phần assistant
  (`train_on_responses_only`), notebook in ra phần không bị mask để kiểm tra bằng mắt.
- DPO: **bẫy model tham chiếu**. DPOTrainer + PEFT + `ref_model=None` lấy model tắt adapter làm tham chiếu. Load
  adapter SFT rồi train tiếp thì "tắt adapter" = Qwen3 gốc chứ không phải SFT. Cách làm: merge SFT vào trọng số
  (16-bit), load lại 4-bit, gắn LoRA mới cho DPO.
- Cặp DPO: nhiễu có chủ đích (rẻ, phủ đều các kiểu lỗi) + lỗi thật của SFT trên 600 câu train (sát phân phối lỗi
  thật hơn). Output hỏng format giữ nguyên chuỗi làm rejected.
- Unsloth cần CUDA capability ≥ 7.0: Kaggle phải chọn T4, P100 (6.0) không chạy.

## Chạy trên laptop

- llama-cpp-python: không dùng `create_chat_completion` vì chat handler không truyền `enable_thinking=False`.
  Tự render chat template lấy từ metadata GGUF bằng jinja2 rồi gọi `create_completion`.
- **Bẫy `tojson`**: filter `tojson` mặc định của jinja2 escape sang `ệ...` (ensure_ascii) và escape HTML.
  transformers override filter này bằng `json.dumps(ensure_ascii=False)`. Không override theo thì prompt lúc chạy
  GGUF khác hẳn lúc train (tiếng Việt trong mô tả tool thành chuỗi \u), model kém đi mà không báo lỗi gì.
  Có test kiểm tra prompt render ra không chứa `\u`.
- `warmup()` prefill sẵn system + tools lúc khởi động để câu đầu tiên không phải chờ.
- tokens/s trong `Action.meta` tính cả thời gian prefill, nên thấp hơn tốc độ decode thuần.

## Việc còn lại

- Chạy notebook 01-04, điền bảng kết quả, đo độ trễ thật trên laptop (Ryzen 5 5625U).
- Thử bỏ bớt tool không dùng trong ngữ cảnh (vd khi đứng yên thì không cần `set_drive_mode`) để rút ngắn prompt.
- Lệnh tương đối ("to lên chút") cần state có giá trị hiện tại của điều hoà/âm lượng.
- Tăng bộ viết tay lên ~150 câu, nhờ người khác nói thử (giọng miền Trung chưa có).
