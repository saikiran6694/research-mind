import json
import time
import os
import re
import random
from typing import Optional
from dotenv import load_dotenv

import google.genai as genai

load_dotenv()

# ── Free tier constants ──────────────────────────────────────────
FREE_TIER_RPM        = 15      # requests per minute (be conservative: use 10)
SAFE_RPM             = 10      # leave headroom below the hard limit
MIN_GAP_SECONDS      = 60.0 / SAFE_RPM   # 6s between calls
MAX_RETRIES          = 6
BASE_BACKOFF         = 30      # seconds for first retry



class RateLimiterLLM:
    """
    Singleton-friendly wrapper round Gemini that:
      - Enforces a minimum gap between calls (RPM control)
      - Retries on ResourceExhausted with exponential backoff + jitter
      - Parses retry-after hints from error messages
      - Tracks approximate daily call count
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", rpm_limit: int = SAFE_RPM):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment.")

        self.client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self.rpm_limit   = rpm_limit
        self.min_gap     = 60.0 / rpm_limit
        self._last_call  = 0.0
        self._call_count = 0
        self._daily_file = ".daily_file.json"
        self._load_daily()

    
    def invoke(self, prompt: str, max_retries: int = MAX_RETRIES) -> str:
        """Send a prompt, respecting rate limits and retrying on quota errors."""
        self._enforce_gap()

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(model=self._model_name, contents=prompt)
                self._last_call = time.time()
                self._call_count += 1
                self._save_daily()
                return response.text
            except Exception as e:
                err = str(e).lower()
                is_quota = any(k in err for k in
                               ["resource_exhausted", "quota", "rate limit",
                                "429", "too many requests"])
                
                if is_quota:
                    wait = self._parse_retry_delay(str(e)) or (BASE_BACKOFF * (2 ** attempt))
                    jitter = random.uniform(0, 10)
                    total_wait = wait + jitter
                    print(f"[RateLimiter] Quota hit — attempt {attempt+1}/{max_retries}. "
                          f"Waiting {total_wait:.0f}s...")
                    time.sleep(total_wait)
                    continue

                if attempt < max_retries - 1:
                    print(f"[RateLimiter] Error: {e}. Retrying in 10s...")
                    time.sleep(10)
                    continue
            
            raise

        raise RuntimeError(f"[RateLimiter] Max retries ({max_retries}) exceeded.")

    @property
    def daily_calls(self) -> int:
        return self._call_count

    def _enforce_gap(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_gap:
            sleep_for = self.min_gap - elapsed
            print(f"[RateLimiter] Throttling — waiting {sleep_for:.1f}s")
            time.sleep(sleep_for)

    @staticmethod
    def _parse_retry_delay(error_msg: str) -> Optional[int]:
        """Try to extract a retry-after value from the error string."""
        patterns = [
            r"retry[_ ]after[:\s]+(\d+)",
            r"(\d+)\s*second",
            r"wait\s+(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, error_msg, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def _load_daily(self):
        from datetime import date
        today = str(date.today())
        try:
            data = json.loads(open(self._daily_file).read())
            if data.get("date") == today:
                self._call_count = data.get("count", 0)
                return
        except Exception:
            pass
        self._call_count = 0

    def _save_daily(self):
        from datetime import date
        with open(self._daily_file, "w") as f:
            json.dump({"date": str(date.today()), "count": self._call_count}, f)
        

# ── Module-level singleton ───────────────────────────────────────\
_llm_instance: Optional[RateLimiterLLM] = None

def get_llm(model_name: str = "gemini-2.5-flash", rpm_limit: int = SAFE_RPM) -> RateLimiterLLM:
    """Return the shared RateLimitedLLM instance (lazy init)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = RateLimiterLLM(model_name=model_name, rpm_limit=rpm_limit)
    return _llm_instance


# prompt = """You are a research planner. 

# Generate exactly 4 precise, diverse web search queries for the topic below. Each query should cover a different angle.

# Topic: Different frameworks used for developing agentic workflows
# Focus areas: general overview
# Existing queries already run: []
# Knowledge gap identified so far: []

# Return ONLY a JSON array of 4 query strings. No explanation. No markdown.
# Example: ["query one", "query two", "query three"]
# """

# model = get_llm()
# response = model.invoke(prompt=prompt)
# print(response)