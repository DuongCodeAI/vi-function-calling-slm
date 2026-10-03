"""Client REST tối giản cho API kiểu OpenAI (Groq, llama.cpp server). Copy và rút gọn từ vn-traffic-law-rag.

Chỉ dùng để SINH dữ liệu, mặc định Groq `openai/gpt-oss-120b` (Apache-2.0, được dùng output để train).
Không dùng Gemini ở đây: điều khoản Gemini API cấm dùng output để phát triển model cạnh tranh.
"""

import logging
import os
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "local": (os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"), None, "local"),
}


class LLMError(RuntimeError):
    pass


class _DailyLimit(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float


def _is_daily_limit(r) -> bool:
    """429 do hết quota NGÀY (đợi hàng giờ) khác 429 do quá nhanh trong phút (đợi vài giây)."""
    wait = float(r.headers.get("retry-after", 0) or 0)
    return wait > 300 or "per day" in r.text.lower() or "(tpd)" in r.text.lower() or "(rpd)" in r.text.lower()


class LLMClient:
    def __init__(self, provider: str = "groq", model: str | None = None, retries: int = 4, timeout: int = 120,
                 fallback_models: list[str] | None = None):
        base, key_env, default_model = PROVIDERS[provider]
        self.base_url = base
        self.key = os.environ[key_env] if key_env else None  # thiếu key thì lỗi ngay, đừng đợi tới lúc gọi
        self.model = model or os.getenv("GROQ_MODEL", default_model)
        # Groq tính quota theo TỪNG model: hết quota ngày của 120b thì chuyển sang model mở khác (cùng 1 key).
        # Chỉ dùng model license mở (Apache-2.0) để output được phép đem đi train.
        env = os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b")
        self.fallbacks = fallback_models if fallback_models is not None else [m for m in env.split(",") if m]
        self.retries = retries
        self.timeout = timeout

    def _next_model(self) -> bool:
        if not self.fallbacks:
            return False
        old, self.model = self.model, self.fallbacks.pop(0)
        log.warning("%s hết quota ngày, chuyển sang %s", old, self.model)
        return True

    def complete(self, system: str, user: str, temperature: float = 0.9, json_mode: bool = True) -> LLMResponse:
        while True:
            try:
                return self._complete_once(system, user, temperature, json_mode)
            except _DailyLimit as e:
                if not self._next_model():
                    raise LLMError(f"hết quota ngày của mọi model: {e}") from e

    def _complete_once(self, system: str, user: str, temperature: float, json_mode: bool) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.model.startswith("openai/gpt-oss"):
            # viết câu ngắn thì không cần suy luận dài, effort thấp nhanh hơn và đỡ tốn quota token/phút
            body["reasoning_effort"] = "low"
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                                  timeout=self.timeout)
            except requests.RequestException as e:
                last = str(e)
            else:
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"].get("content") or ""
                    return LLMResponse(text, self.model, time.perf_counter() - t0)
                last = f"{r.status_code}: {r.text[:200]}"
                if r.status_code == 429 and _is_daily_limit(r):
                    raise _DailyLimit(last)
                if r.status_code == 429:
                    # Groq free tier giới hạn theo phút, header retry-after cho biết phải đợi bao lâu
                    wait = float(r.headers.get("retry-after", 10 * (attempt + 1)))
                    log.warning("429, đợi %.0fs", wait)
                    time.sleep(min(wait, 120))
                    continue
                if r.status_code < 500:
                    break  # lỗi 4xx khác (sai body, hết quota ngày) thì thử lại cũng vô ích
            log.warning("LLM lỗi %s, thử lại", last)
            time.sleep(2 * (attempt + 1))
        raise LLMError(last)
