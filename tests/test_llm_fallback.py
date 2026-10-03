import pytest

from vi_fc import llm


class _Resp:
    def __init__(self, status, text="", headers=None, content=None):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _client(monkeypatch, responses):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(json["model"])
        return responses.pop(0)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return llm.LLMClient(fallback_models=["openai/gpt-oss-20b"]), calls


def test_switch_model_on_daily_limit(monkeypatch):
    c, calls = _client(monkeypatch, [
        _Resp(429, "Rate limit reached ... tokens per day (TPD)", {"retry-after": "3600"}),
        _Resp(200, content='{"ok": 1}'),
    ])
    r = c.complete("s", "u")
    assert calls == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert r.model == "openai/gpt-oss-20b" and c.model == "openai/gpt-oss-20b"


def test_minute_limit_waits_same_model(monkeypatch):
    c, calls = _client(monkeypatch, [_Resp(429, "per minute", {"retry-after": "5"}), _Resp(200, content="{}")])
    c.complete("s", "u")
    assert calls == ["openai/gpt-oss-120b", "openai/gpt-oss-120b"]


def test_all_models_exhausted(monkeypatch):
    daily = _Resp(429, "tokens per day", {"retry-after": "7200"})
    c, _ = _client(monkeypatch, [daily, daily])
    with pytest.raises(llm.LLMError, match="mọi model"):
        c.complete("s", "u")
