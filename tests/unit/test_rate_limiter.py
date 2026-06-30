import time
import pytest
from unittest.mock import MagicMock, patch, call

import utils.rate_limiter as rl_module
from utils.rate_limiter import RateLimiterLLM, get_llm


@pytest.fixture(autouse=True)
def reset_singleton():
    rl_module._llm_instance = None
    yield
    rl_module._llm_instance = None


@pytest.fixture
def mock_genai_client():
    mock_response = MagicMock()
    mock_response.text = "mocked LLM response"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


@pytest.fixture
def llm(mock_genai_client):
    with patch("utils.rate_limiter.genai.Client", return_value=mock_genai_client):
        instance = RateLimiterLLM(model_name="gemini-test", rpm_limit=60)
        instance._last_call = 0.0
        instance._call_count = 0
        return instance


class TestParseRetryDelay:
    def test_retry_after_pattern(self):
        assert RateLimiterLLM._parse_retry_delay("retry after 30 seconds") == 30

    def test_retry_underscore_pattern(self):
        assert RateLimiterLLM._parse_retry_delay("retry_after: 60") == 60

    def test_seconds_pattern(self):
        assert RateLimiterLLM._parse_retry_delay("please wait 45 seconds") == 45

    def test_wait_pattern(self):
        assert RateLimiterLLM._parse_retry_delay("wait 120 before retrying") == 120

    def test_no_match_returns_none(self):
        assert RateLimiterLLM._parse_retry_delay("quota exceeded") is None

    def test_case_insensitive(self):
        assert RateLimiterLLM._parse_retry_delay("RETRY AFTER 90") == 90

    def test_extracts_first_match(self):
        result = RateLimiterLLM._parse_retry_delay("retry after 10 seconds, then 20")
        assert result in (10, 20)


class TestEnforceGap:
    def test_sleeps_when_called_too_soon(self, llm):
        llm._last_call = time.time()
        with patch("time.sleep") as mock_sleep:
            llm._enforce_gap()
            mock_sleep.assert_called_once()
            sleep_duration = mock_sleep.call_args[0][0]
            assert sleep_duration > 0

    def test_no_sleep_when_gap_is_sufficient(self, llm):
        llm._last_call = time.time() - 100.0
        with patch("time.sleep") as mock_sleep:
            llm._enforce_gap()
            mock_sleep.assert_not_called()

    def test_sleep_duration_covers_remaining_gap(self, llm):
        llm.min_gap = 6.0
        llm._last_call = time.time() - 2.0
        with patch("time.sleep") as mock_sleep:
            llm._enforce_gap()
            sleep_duration = mock_sleep.call_args[0][0]
            assert 3.5 <= sleep_duration <= 4.5


class TestInvoke:
    def test_returns_response_text(self, llm):
        with patch("time.sleep"):
            result = llm.invoke("What is AI?")
        assert result == "mocked LLM response"

    def test_increments_daily_calls(self, llm):
        initial = llm.daily_calls
        with patch("time.sleep"), patch.object(llm, "_save_daily"):
            llm.invoke("prompt")
        assert llm.daily_calls == initial + 1

    def test_multiple_calls_increment_counter(self, llm):
        with patch("time.sleep"), patch.object(llm, "_save_daily"):
            llm.invoke("prompt 1")
            llm.invoke("prompt 2")
            llm.invoke("prompt 3")
        assert llm.daily_calls == 3

    def test_retries_on_quota_error(self, llm, mock_genai_client):
        mock_response = MagicMock()
        mock_response.text = "success after retry"
        mock_genai_client.models.generate_content.side_effect = [
            Exception("resource_exhausted: quota exceeded"),
            mock_response,
        ]
        with patch("time.sleep"), patch.object(llm, "_save_daily"):
            result = llm.invoke("prompt", max_retries=3)
        assert result == "success after retry"
        assert mock_genai_client.models.generate_content.call_count == 2

    def test_retries_on_429_error(self, llm, mock_genai_client):
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_genai_client.models.generate_content.side_effect = [
            Exception("429 too many requests"),
            mock_response,
        ]
        with patch("time.sleep"), patch.object(llm, "_save_daily"):
            result = llm.invoke("prompt", max_retries=3)
        assert result == "ok"

    def test_raises_after_max_retries(self, llm, mock_genai_client):
        mock_genai_client.models.generate_content.side_effect = Exception("resource_exhausted")
        with patch("time.sleep"), pytest.raises(Exception):
            llm.invoke("prompt", max_retries=2)

    def test_non_quota_error_retries_then_raises(self, llm, mock_genai_client):
        mock_genai_client.models.generate_content.side_effect = Exception("network timeout")
        with patch("time.sleep"), pytest.raises(Exception):
            llm.invoke("prompt", max_retries=2)


class TestGetLlmSingleton:
    def test_returns_rate_limiter_instance(self):
        with patch("utils.rate_limiter.RateLimiterLLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            instance = get_llm()
            assert instance is not None

    def test_returns_same_instance_on_repeated_calls(self):
        with patch("utils.rate_limiter.RateLimiterLLM") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            a = get_llm()
            b = get_llm()
            assert a is b

    def test_constructor_called_once(self):
        with patch("utils.rate_limiter.RateLimiterLLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm()
            get_llm()
            get_llm()
            mock_cls.assert_called_once()


class TestDailyCallTracking:
    def test_load_daily_resets_on_new_day(self, tmp_path, mock_genai_client):
        daily_file = tmp_path / ".daily_file.json"
        daily_file.write_text('{"date": "2000-01-01", "count": 99}')

        with patch("utils.rate_limiter.genai.Client", return_value=mock_genai_client):
            instance = RateLimiterLLM.__new__(RateLimiterLLM)
            instance._daily_file = str(daily_file)
            instance._load_daily()
            assert instance._call_count == 0

    def test_load_daily_restores_count_for_same_day(self, tmp_path, mock_genai_client):
        from datetime import date
        daily_file = tmp_path / ".daily_file.json"
        daily_file.write_text(json_dump({"date": str(date.today()), "count": 17}))

        with patch("utils.rate_limiter.genai.Client", return_value=mock_genai_client):
            instance = RateLimiterLLM.__new__(RateLimiterLLM)
            instance._daily_file = str(daily_file)
            instance._load_daily()
            assert instance._call_count == 17


def json_dump(obj):
    import json
    return json.dumps(obj)
