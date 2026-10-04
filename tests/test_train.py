from vi_fc.train import plan_steps, tokenize_sft


class CharTok:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def test_tokenize_sft_masks_prompt():
    row = tokenize_sft(CharTok(), "abc", "xy", 10)
    assert row["input_ids"] == [97, 98, 99, 120, 121]
    assert row["labels"] == [-100, -100, -100, 120, 121]
    assert len(row["attention_mask"]) == 5


def test_tokenize_sft_drops_too_long():
    assert tokenize_sft(CharTok(), "abc", "xy", 4) is None


def test_plan_steps():
    # ít dữ liệu: đủ 2 epoch
    assert plan_steps(160, 16, 2, 1700, 1500, 40) == 20
    # nhiều dữ liệu: bị ngân sách chặn, 40 phút * 1500 tok/s / (16 * 1700) = 132
    assert plan_steps(2000, 16, 2, 1700, 1500, 40) == 132
    assert plan_steps(1, 16, 1, 10**9, 1, 1) == 1
