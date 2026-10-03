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


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float


class LLMClient:
    def __init__(self, provider: str = "groq", model: str | None = None, retries: int = 4, timeout: int = 120):
        base, key_env, default_model = PROVIDERS[provider]
        self.base_url = base
        self.key = os.environ[key_env] if key_env else None  # thiếu key thì lỗi ngay, đừng đợi tới lúc gọi
        self.model = model or os.getenv("GROQ_MODEL", default_model)
        self.retries = retries
        self.timeout = timeout

    def complete(self, system: str, user: str, temperature: float = 0.9, json_mode: bool = True) -> LLMResponse:
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
