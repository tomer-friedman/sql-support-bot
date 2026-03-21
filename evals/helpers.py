"""Shared helpers for the eval suite."""

import time
from langchain_core.messages import AIMessage


def get_tool_names(messages: list) -> list[str]:
    """Return all tool names called across all AIMessages."""
    names = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            names.extend(tc["name"] for tc in msg.tool_calls)
    return names


def get_final_response(messages: list) -> str:
    """Return the content of the last AIMessage with non-empty content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def get_tool_call_args(messages: list, tool_name: str) -> dict:
    """Return the args dict of the FIRST call to tool_name, or {} if not found."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == tool_name:
                    return tc.get("args", {})
    return {}


# ---------------------------------------------------------------------------
# Rate control
# ---------------------------------------------------------------------------

# Proactive throttle: minimum seconds between consecutive *real* API calls.
# At ~2-3K tokens per call and a 30K TPM limit we can safely do ~10 calls/min,
# so 2 s spacing keeps us comfortably under the limit without ever needing to
# wait for a 429. Increase this value if you have a lower TPM tier.
#
# Cache hits are excluded from throttling: they return in <50 ms and consume
# no tokens. Only calls that exceed _CACHE_HIT_THRESHOLD update the timer.
_MIN_INVOKE_INTERVAL: float = 2.0
_CACHE_HIT_THRESHOLD: float = 0.5  # seconds; real API calls always exceed this
_last_api_call_time: float = 0.0


def invoke_agent(
    agent,
    messages: list,
    max_retries: int = 2,
    retry_delay: float = 10.0,
) -> dict:
    """
    Throttled agent invocation with two layers of rate-limit protection:

    1. Proactive throttle — enforces a minimum gap between *real* API calls so
       we stay within the TPM limit by default and never trigger a 429. Cache
       hits are excluded: they consume no tokens so no throttle is applied.
    2. Retry with backoff — a safety net for unexpected bursts. Uses a fixed
       delay because the OpenAI error message tells us exactly how long to wait.

    LangSmith tracing is handled by the @pytest.mark.langsmith plugin.
    """
    global _last_api_call_time
    from openai import RateLimitError

    elapsed = time.monotonic() - _last_api_call_time
    if elapsed < _MIN_INVOKE_INTERVAL:
        time.sleep(_MIN_INVOKE_INTERVAL - elapsed)

    for attempt in range(max_retries + 1):
        try:
            call_start = time.monotonic()
            result = agent.invoke({"messages": messages})
            # Only track timing for real API calls; cache hits are near-instant.
            if time.monotonic() - call_start > _CACHE_HIT_THRESHOLD:
                _last_api_call_time = time.monotonic()
            return result
        except RateLimitError:
            if attempt == max_retries:
                raise
            print(f"\n[rate limit] unexpected 429 — waiting {retry_delay}s before retry {attempt + 2}/{max_retries + 1}…")
            time.sleep(retry_delay)

    raise RuntimeError("invoke_agent: exhausted retries")


# ---------------------------------------------------------------------------
# LLM-as-judge (via openevals)
# ---------------------------------------------------------------------------

def run_llm_judge(
    llm_judge_client,
    criteria: str,
    user_input: str,
    agent_response: str,
) -> tuple[bool, str]:
    """
    Call the openevals judge and return (passed, reasoning).

    Errors propagate — a broken judge configuration should fail the test
    visibly rather than silently defaulting to a pass.
    """
    result = llm_judge_client(
        user_input=user_input,
        agent_response=agent_response,
        criteria=criteria,
    )
    score = result.get("score")
    reasoning = result.get("reasoning") or result.get("comment", "")
    return bool(score), reasoning
