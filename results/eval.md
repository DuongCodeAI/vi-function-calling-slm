# Kết quả đánh giá

Sinh tự động bởi `python -m vi_fc.evaluate report`.

## Model (không guard)

| system | dataset | n | kind | tool | args exact | arg F1 | multi | tool bịa | ask P | ask R | refuse R | từ chối thừa | format | p50 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b-base | manual | 45 | 0.733 | 0.788 | 0.515 | 0.658 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 1.000 | 1.870 |
| qwen3-1.7b-base | synth_test | 30 | 0.700 | 0.850 | 0.550 | 0.764 | 0.333 | 0.000 | - | 0.000 | - | 0.000 | 1.000 | 2.456 |
| sft | manual | 45 | 0.756 | 0.970 | 0.909 | 0.931 | 0.500 | 0.000 | - | 0.000 | 0.000 | 0.024 | 1.000 | 1.808 |
| sft | synth_test | 30 | 0.733 | 1.000 | 0.800 | 0.885 | 0.667 | 0.000 | 1.000 | 0.143 | - | 0.067 | 1.000 | 1.976 |
| sft-dpo | manual | 45 | 0.800 | 0.909 | 0.849 | 0.886 | 0.500 | 0.000 | 0.500 | 0.600 | 0.333 | 0.024 | 1.000 | 1.831 |
| sft-dpo | synth_test | 30 | 0.667 | 0.950 | 0.800 | 0.902 | 0.667 | 0.000 | 0.333 | 0.143 | - | 0.067 | 1.000 | 1.954 |
| sft-dpo-q4km | manual | 30 | 0.967 | 0.933 | 0.867 | 0.906 | 0.667 | 0.000 | - | - | - | 0.000 | 1.000 | 7.546 |

## Model + guard luật an toàn

Lưu ý: nhãn refuse của tập tổng hợp cũng sinh từ chính các luật này, nên refuse R sau guard gần như chắc chắn bằng 1. Con số đáng xem là cột không guard và từ chối thừa.

| system | dataset | n | exact | refuse R | gọi tool không an toàn | từ chối thừa |
|---|---|---|---|---|---|---|
| qwen3-1.7b-base | manual | 45 | 0.489 | 1.000 | 0.000 | 0.000 |
| qwen3-1.7b-base | synth_test | 30 | 0.400 | - | - | 0.033 |
| sft | manual | 45 | 0.800 | 1.000 | 0.000 | 0.024 |
| sft | synth_test | 30 | 0.600 | - | - | 0.067 |
| sft-dpo | manual | 45 | 0.756 | 0.667 | 0.000 | 0.024 |
| sft-dpo | synth_test | 30 | 0.567 | - | - | 0.067 |
| sft-dpo-q4km | manual | 30 | 0.867 | - | - | 0.000 |
