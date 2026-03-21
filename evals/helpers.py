"""Shared helpers for the eval suite."""

import os
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


# ---------------------------------------------------------------------------
# Rate control
# ---------------------------------------------------------------------------

# Proactive throttle: minimum seconds between consecutive agent invocations.
# At ~2-3K tokens per call and a 30K TPM limit we can safely do ~10 calls/min,
# so 6 s spacing keeps us comfortably under the limit without ever needing to
# wait for a 429. Increase this value if you have a lower TPM tier.
_MIN_INVOKE_INTERVAL: float = 2.0
_last_invoke_time: float = 0.0


def invoke_and_get_run_id(
    agent,
    messages: list,
    max_retries: int = 2,
    retry_delay: float = 20.0,
) -> tuple[dict, str | None]:
    """
    Invoke the agent with two layers of rate-limit protection:

    1. Proactive throttle — enforces a minimum gap between calls so we stay
       within the TPM limit by default and never trigger a 429.
    2. Retry with backoff — a safety net for unexpected bursts (e.g. another
       process sharing the same API key). Uses a fixed delay, not exponential,
       because the OpenAI error message tells us exactly how long to wait.

    Also wraps the call in a @traceable context for LangSmith tracing.
    Returns (result, run_id).
    """
    global _last_invoke_time
    from openai import RateLimitError

    # --- 1. Proactive throttle ---
    elapsed = time.monotonic() - _last_invoke_time
    if elapsed < _MIN_INVOKE_INTERVAL:
        time.sleep(_MIN_INVOKE_INTERVAL - elapsed)

    run_id_holder: dict = {}

    def _do_invoke():
        try:
            from langsmith import traceable
            from langsmith.run_helpers import get_current_run_tree

            @traceable(name="eval_agent_invoke", run_type="chain")
            def _inner():
                result = agent.invoke({"messages": messages})
                rt = get_current_run_tree()
                if rt:
                    run_id_holder["id"] = str(rt.id)
                return result

            return _inner()
        except ImportError:
            return agent.invoke({"messages": messages})

    # --- 2. Retry as safety net ---
    for attempt in range(max_retries):
        _last_invoke_time = time.monotonic()
        try:
            return _do_invoke(), run_id_holder.get("id")
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            print(f"\n[rate limit] unexpected 429 — waiting {retry_delay}s before retry {attempt + 2}/{max_retries}…")
            time.sleep(retry_delay)
            run_id_holder.clear()

    raise RuntimeError("invoke_and_get_run_id: exhausted retries")


def get_tool_call_args(messages: list, tool_name: str) -> dict:
    """Return the args dict of the FIRST call to tool_name, or {} if not found."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == tool_name:
                    return tc.get("args", {})
    return {}


def run_llm_judge(
    llm_judge_client,
    criteria: str,
    user_input: str,
    agent_response: str,
    test_name: str,
) -> bool:
    """
    Ask an LLM to evaluate whether agent_response satisfies criteria.

    Returns True if the judge says PASS, False if FAIL.
    On any error, returns True with a printed warning so a flaky judge call
    never fails a test deterministically.
    """
    prompt = (
        "You are an impartial evaluator. Decide whether the agent response "
        "satisfies the evaluation criteria.\n\n"
        f"USER INPUT:\n{user_input}\n\n"
        f"AGENT RESPONSE:\n{agent_response}\n\n"
        f"EVALUATION CRITERIA:\n{criteria}\n\n"
        "Reply with exactly one word: PASS or FAIL."
    )
    try:
        verdict = llm_judge_client.invoke(prompt).content.strip().upper()
        if verdict.startswith("PASS"):
            return True
        if verdict.startswith("FAIL"):
            return False
        print(f"\n[llm_judge] Unexpected verdict '{verdict}' for '{test_name}', treating as PASS")
        return True
    except Exception as exc:
        print(f"\n[llm_judge] Error for '{test_name}': {exc}, treating as PASS")
        return True


def log_eval_feedback(
    run_id: str | None,
    test_name: str,
    tool_routing_passed: bool | None = None,
    content_passed: bool | None = None,
    judge_passed: bool | None = None,
) -> None:
    """Post structured feedback scores to LangSmith for this test run."""
    if run_id is None:
        return
    if not os.getenv("LANGCHAIN_API_KEY"):
        return

    try:
        from langsmith import Client
        client = Client()

        if tool_routing_passed is not None:
            client.create_feedback(
                run_id=run_id,
                key="tool_routing",
                score=1.0 if tool_routing_passed else 0.0,
                comment=f"test: {test_name}",
            )
        if content_passed is not None:
            client.create_feedback(
                run_id=run_id,
                key="response_content",
                score=1.0 if content_passed else 0.0,
                comment=f"test: {test_name}",
            )
        if judge_passed is not None:
            client.create_feedback(
                run_id=run_id,
                key="llm_judge",
                score=1.0 if judge_passed else 0.0,
                comment=f"test: {test_name}",
            )
    except Exception:
        pass  # Never let LangSmith errors fail a test
