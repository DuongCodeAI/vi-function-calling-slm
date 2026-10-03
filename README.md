# vi-function-calling-slm

Dạy một model nhỏ (Qwen3-1.7B) hiểu lệnh tiếng Việt trong xe và gọi đúng hàm, chạy **offline trên CPU laptop**
bằng GGUF Q4_K_M (~1.1GB).

Hành vi mong muốn (lấy từ bộ test viết tay):

```
[xe đang chạy 60 km/h]  "nóng quá, cho máy lạnh xuống 22 độ đi em"
→ <tool_call>{"name": "set_climate", "arguments": {"temperature": 22}}</tool_call>

[xe đang chạy 30 km/h]  "mở khoá cửa đi, để bạn anh lên"
→ "Xe đang chạy 30 km/h, mở khoá cửa lúc này không an toàn. Bạn dừng hẳn xe rồi mình mở nhé."

"gọi điện giúp anh cái"  → "Bạn muốn gọi cho ai?"
```

Đây là phần "hiểu lệnh" của [viet-copilot](https://github.com/DuongCodeAI/viet-copilot). Hai tool
`lookup_traffic_law` / `lookup_sign` gọi sang [vn-traffic-law-rag](https://github.com/DuongCodeAI/vn-traffic-law-rag).

## Ý tưởng

Model lớn gọi hàm tốt nhưng cần mạng; trong xe thì mạng chập chờn và mỗi lệnh "bật điều hoà" không đáng một lần
gọi API. Mình chưng cất (distill) khả năng đó sang model 1.7B:

```
seed do code bốc (kịch bản, tool, giá trị, trạng thái xe, văn phong)
  ─► gpt-oss-120b (Groq) chỉ viết CÂU NGƯỜI DÙNG
  ─► nhãn gold dựng từ seed bằng code (schema + luật an toàn)
  ─► lọc: câu có nhắc đủ giá trị? bỏ trùng (3-gram Jaccard) ─► chia train/val/test theo seed
  ─► SFT QLoRA (Unsloth, T4) ─► DPO với cặp lỗi điển hình ─► GGUF q4_k_m ─► llama.cpp trên laptop
                                                                                 └► guard luật an toàn
```

- **Nhãn không do LLM gán.** LLM chỉ viết câu; đáp án (gọi hàm gì, tham số gì, hay phải hỏi lại/từ chối)
  sinh từ seed nên luôn đúng schema và đúng luật. Rủi ro còn lại là LLM viết lệch seed → có bước kiểm tra
  câu có nhắc đủ giá trị (22 độ, "ghế phụ", "Hồ Gươm"...) và bỏ câu không khớp.
- **4 loại kết quả**: gọi hàm (1 hoặc nhiều), hỏi lại khi thiếu thông tin, từ chối khi không an toàn với trạng
  thái xe, trả lời thường khi ngoài phạm vi.
- **Văn phong**: văn nói, chen tiếng Anh ("bật AC"), giọng Bắc/Nam, gõ không dấu, gõ sai (hai cái cuối làm bằng
  code chứ không nhờ LLM).
- **Giữ nguyên chat template của Qwen3** (`tools=`, `<tool_call>`, tắt thinking): model không phải học lại cú pháp.
- **DPO** với rejected là lỗi hay gặp: sai tool, bịa tên tool, 220 thay vì 22 độ, đoán bừa thay vì hỏi lại,
  làm theo yêu cầu không an toàn... cộng lỗi thật của model SFT.
- **Phòng thủ 2 lớp**: model học từ chối, nhưng sau output vẫn có guard kiểm tra schema và luật an toàn.

## Kết quả

**Chưa chạy.** Code và notebook đã xong, test local pass; phần sinh dữ liệu + train cần chạy trên Kaggle.
Bảng dưới sẽ điền từ `results/eval.md` sau khi chạy notebook 04, không điền số ước đoán.

| system | kind acc | tool acc | args exact | ask R | refuse R | từ chối thừa | p50 (laptop) |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B gốc (prompt) | chưa chạy | | | | | | |
| + SFT | chưa chạy | | | | | | |
| + SFT + DPO | chưa chạy | | | | | | |
| + SFT + DPO, GGUF Q4_K_M | chưa chạy | | | | | | |
| Gemini (cận trên, chỉ để so) | chưa chạy | | | | | | |

Hai tập test: phần test của dữ liệu tổng hợp (chia theo seed, không trùng yêu cầu với train) và
`data/test_manual.jsonl` — 45 câu mình tự viết theo cách mình nói thật trong xe, có đáp án viết tay.

Metric (`vi_fc/evaluate.py`): đúng loại kết quả, đúng tool, đúng toàn bộ tham số, F1 theo tham số, tỉ lệ tool
bịa, precision/recall của hỏi lại, recall từ chối + tỉ lệ vẫn gọi tool khi lẽ ra phải từ chối, tỉ lệ từ chối thừa,
khớp đủ các lệnh trong câu nhiều lệnh, độ trễ.

## Chạy thử

```bash
pip install -e ".[infer]"
# tải GGUF từ HF Hub: DuongCodeAI/vi-fc-qwen3-1.7b-GGUF (sau khi chạy notebook 04)
python -m vi_fc.inference "bật điều hoà 22 độ ghế phụ" --model models/vi-fc-qwen3-1.7b-Q4_K_M.gguf --speed 40
```

```python
from vi_fc.inference import FunctionCaller

fc = FunctionCaller("llama_cpp", "models/vi-fc-qwen3-1.7b-Q4_K_M.gguf", n_threads=6)
fc.warmup()  # prefill sẵn system + tools
a = fc.decide("mở khoá cửa", {"speed_kmh": 40, "gear": "D"})
a.kind, a.calls, a.text   # "refuse", [], "Xe đang chạy 40 km/h, ..."
```

Không có GGUF thì dùng backend `"openai"` trỏ vào llama.cpp server / API bất kỳ kiểu OpenAI.

Sinh dữ liệu và đánh giá (cần `GROQ_API_KEY`):

```bash
python -m vi_fc.datagen.generate --n 4000 --out data/gen/raw.jsonl
python -m vi_fc.datagen.filters --raw data/gen/raw.jsonl --out data/gen
python -m vi_fc.dpo --data data/gen/train.jsonl --out data/gen/dpo_train.jsonl
python -m vi_fc.evaluate predict --data data/test_manual.jsonl --dataset manual --system q4 \
    --backend llama_cpp --model models/vi-fc-qwen3-1.7b-Q4_K_M.gguf --out results/preds/q4_manual.jsonl
python -m vi_fc.evaluate report --preds results/preds --out results
```

## Notebook (Kaggle hoặc Colab, GPU T4)

| notebook | việc | phần cứng |
|---|---|---|
| `01_generate_data` | sinh + lọc dữ liệu, cặp DPO | CPU |
| `02_sft_qlora` | SFT QLoRA r=16, 2 epoch, chỉ tính loss phần assistant | T4 |
| `03_dpo` | DPO beta=0.1 trên bản SFT đã merge + lỗi on-policy | T4 |
| `04_export_eval` | GGUF q4_k_m, đánh giá 5 hệ thống | T4 + CPU |

Unsloth cần GPU có CUDA capability ≥ 7.0: T4 được, P100 thì không.

Chạy trên Colab: mở notebook từ GitHub (File → Open notebook → GitHub → DuongCodeAI/vi-function-calling-slm), chọn T4 GPU,
thêm Secrets `GROQ_API_KEY`, `HF_TOKEN`, `GEMINI_API_KEY` (tuỳ chọn). Dữ liệu, adapter, bản merge và checkpoint lưu trên
Google Drive (`MyDrive/ai-portfolio/vi-function-calling-slm`, cần ~10GB trống), bị ngắt thì chạy lại notebook là train tiếp.

## Hạn chế

- 17 tool và 3 luật an toàn do mình tự đặt để minh hoạ, không phải API/đặc tả của hãng xe nào.
- Chưa hỗ trợ lệnh tương đối ("to lên chút", "mát thêm xíu") — cần biết giá trị hiện tại, để sau.
- Phân loại hỏi lại / từ chối / trả lời dựa vào từ khoá (xem `parse.py`); model đã fine-tune học đúng quy ước
  này, còn model gốc và Gemini có thể bị chấm thiệt một chút ở cột kind.
- Dữ liệu train là tổng hợp; bộ viết tay 45 câu còn nhỏ, chỉ đủ để thấy xu hướng.

## Cấu trúc

```
src/vi_fc/    tools (schema + validator), state, safety, prompt, parse, match, llm (Groq),
              datagen/ (seeds, generate, filters), dpo, evaluate, inference
data/         test_manual.jsonl (viết tay)       notebooks/  Kaggle 01-04
```
