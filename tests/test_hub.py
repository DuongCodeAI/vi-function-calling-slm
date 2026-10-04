from vi_fc.hub import push, repo_id


def test_repo_id_uses_hf_user(monkeypatch):
    monkeypatch.setenv("HF_USER", "someone")
    assert repo_id("vi-fc-qwen3-1.7b") == "someone/vi-fc-qwen3-1.7b"
    assert repo_id("org/vi-fc-synth") == "org/vi-fc-synth"


def test_push_without_token_is_skipped(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert push(str(tmp_path), "vi-fc-qwen3-1.7b") is None
