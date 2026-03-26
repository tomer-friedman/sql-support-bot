"""Shared helpers for the eval suite."""

import re
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

# No proactive throttle — API latency provides natural spacing between calls.
# The retry layer below handles any 429s that do occur.


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True for any 429 / rate-limit error, however the exception is wrapped."""
    current: BaseException | None = exc
    while current is not None:
        try:
            from openai import RateLimitError
            if isinstance(current, RateLimitError):
                return True
        except ImportError:
            pass
        msg = str(current).lower()
        if "rate limit" in msg or "429" in msg or "too many requests" in msg:
            return True
        current = current.__cause__ or current.__context__
    return False


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract the suggested wait time from an OpenAI 429 error message.
    The message contains 'try again in 2.536s' or 'try again in 748ms'."""
    msg = str(exc)
    m = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
    if m:
        return float(m.group(1))
    m = re.search(r"try again in (\d+)ms", msg)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def invoke_agent(
    agent,
    messages: list,
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    """
    Agent invocation with smart retry on rate limit errors.

    Uses the wait time suggested by the API ('try again in Xs') plus a small
    buffer. Falls back to exponential backoff if no hint is given.

    LangSmith tracing is handled by the @pytest.mark.langsmith plugin.
    Per-run trace metadata and tags can be passed through LangChain config.
    """
    for attempt in range(max_retries + 1):
        try:
            return agent.invoke(
                {"messages": messages},
                config={
                    "metadata": metadata or {},
                    "tags": tags or [],
                },
            )
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == max_retries:
                raise
            api_wait = _parse_retry_after(exc)
            if api_wait is not None:
                # API tells us exactly when the bucket refills; add a buffer
                delay = api_wait + 2.0
            else:
                delay = retry_delay * (2 ** attempt)
            print(f"\n[rate limit] 429 — waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})…")
            time.sleep(delay)

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
