# ── Gemini free tier limits ──────────────────────────────────────
GEMINI_MODEL        = "gemini-2.5-flash"
GEMINI_SAFE_RPM     = 10     # actual limit is 15 — stay 33% below
GEMINI_MAX_RETRIES  = 6
GEMINI_BASE_BACKOFF = 30     # seconds before first retry

# ── Per-run source caps ──────────────────────────────────────────
MAX_SOURCES = {
    "shallow": 3,
    "medium":  5,
    "deep":    7,
}

# ── Per-run iteration caps ───────────────────────────────────────
MAX_ITERATIONS = {
    "shallow": 1,   # 3 LLM calls
    "medium":  2,   # 5 LLM calls
    "deep":    3,   # 7 LLM calls
}

# ── Token budgets (chars, ~4 chars per token) ────────────────────
PROMPT_BUDGET = {
    "plan_searches":        2_000,
    "analyze_and_critique": 3_500,
    "synthesize_report":    3_000,
}

# ── Fix 6: daily job guard ───────────────────────────────────────
MAX_LLM_CALLS_PER_DAY = 40   # well below free tier RPD of 1500
