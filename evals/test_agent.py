"""
Unified agent eval test.

Each case in test_cases.json is tested once, covering:
  1. Deterministic tool routing checks (expected calls, arg values, forbidden tools, order, count)
  2. Deterministic content checks (required/forbidden keywords)
  3. LLM-as-judge for subjective criteria (if "llm_judge" is set in the case)

Results are logged to LangSmith via @pytest.mark.langsmith when LANGCHAIN_API_KEY is set.
Without the key, t.log_* calls are no-ops and tests run and assert normally.
"""

import json
import pytest
from pathlib import Path
from langsmith import testing as t

from evals.helpers import (
    get_tool_names,
    get_tool_call_args,
    get_final_response,
    invoke_agent,
    run_llm_judge,
)

_CASES_PATH = Path(__file__).parent / "test_cases.json"


def _load_cases():
    with open(_CASES_PATH) as f:
        return json.load(f)


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = _load_cases()
        if metafunc.config.getoption("--smoke", default=False):
            cases = [c for c in cases if c.get("smoke")]
        metafunc.parametrize("case", cases, ids=[c["name"] for c in cases])


@pytest.mark.langsmith
def test_agent(agent, llm_judge_client, case):
    inp = case["input"]
    messages = [{"role": "user", "content": inp}] if isinstance(inp, str) else inp

    t.log_inputs({"input": inp, "category": case["category"]})

    result = invoke_agent(agent, messages)
    all_messages = result["messages"]
    called_tools = get_tool_names(all_messages)
    response = get_final_response(all_messages)

    t.log_outputs({"response": response, "tools_called": called_tools})
    if case.get("llm_judge"):
        t.log_reference_outputs({"criteria": case["llm_judge"]["criteria"]})

    routing_failures: list[str] = []
    content_failures: list[str] = []
    judge_passed = None  # None means "not evaluated"

    # ------------------------------------------------------------------
    # 1. Expected tools called (and arg substring checks)
    # ------------------------------------------------------------------
    for expected in case.get("expected_tool_calls", []):
        tool_name = expected["tool"]
        if tool_name not in called_tools:
            routing_failures.append(
                f"[{case['name']}] Expected tool '{tool_name}' was not called. "
                f"Actual calls: {called_tools}"
            )
            continue

        if "args" in expected:
            actual_args = get_tool_call_args(all_messages, tool_name)
            for arg_key, expected_substr in expected["args"].items():
                actual_val = str(actual_args.get(arg_key, ""))
                if expected_substr.lower() not in actual_val.lower():
                    routing_failures.append(
                        f"[{case['name']}] Tool '{tool_name}' arg '{arg_key}': "
                        f"expected substring '{expected_substr}' not found in '{actual_val}'"
                    )

    # ------------------------------------------------------------------
    # 2. Forbidden tools
    # ------------------------------------------------------------------
    for tool_name in case.get("forbidden_tools", []):
        if tool_name in called_tools:
            routing_failures.append(
                f"[{case['name']}] Forbidden tool '{tool_name}' was called. "
                f"Actual calls: {called_tools}"
            )

    # ------------------------------------------------------------------
    # 3. Tool call ordering
    # ------------------------------------------------------------------
    if case.get("check_order") and len(case.get("expected_tool_calls", [])) > 1:
        expected_sequence = [e["tool"] for e in case["expected_tool_calls"]]
        filtered = [tool for tool in called_tools if tool in expected_sequence]
        if filtered != expected_sequence:
            routing_failures.append(
                f"[{case['name']}] Tool call order mismatch. "
                f"Expected: {expected_sequence}, got: {filtered}"
            )

    # ------------------------------------------------------------------
    # 4. Max tool calls
    # ------------------------------------------------------------------
    max_calls = case.get("max_tool_calls")
    if max_calls is not None and len(called_tools) > max_calls:
        routing_failures.append(
            f"[{case['name']}] Too many tool calls: {len(called_tools)} > {max_calls}. "
            f"Calls: {called_tools}"
        )

    # ------------------------------------------------------------------
    # 5. Non-empty response
    # ------------------------------------------------------------------
    if case.get("response_must_be_nonempty", True) and not response:
        content_failures.append(f"[{case['name']}] Agent returned an empty response.")

    # ------------------------------------------------------------------
    # 6. Required keywords
    # ------------------------------------------------------------------
    for kw in case.get("required_keywords", []):
        if kw.lower() not in response.lower():
            content_failures.append(
                f"[{case['name']}] Required keyword '{kw}' not found in response.\n"
                f"Response: {response}"
            )

    # ------------------------------------------------------------------
    # 7. Forbidden keywords
    # ------------------------------------------------------------------
    for kw in case.get("forbidden_keywords", []):
        if kw.lower() in response.lower():
            content_failures.append(
                f"[{case['name']}] Forbidden keyword '{kw}' found in response.\n"
                f"Response: {response}"
            )

    # ------------------------------------------------------------------
    # 8. LLM-as-judge (subjective checks)
    # ------------------------------------------------------------------
    judge_spec = case.get("llm_judge")
    if judge_spec:
        user_input_text = inp if isinstance(inp, str) else inp[-1]["content"]
        judge_passed, reasoning = run_llm_judge(
            llm_judge_client=llm_judge_client,
            criteria=judge_spec["criteria"],
            user_input=user_input_text,
            agent_response=response,
        )
        if not judge_passed:
            content_failures.append(
                f"[{case['name']}] LLM judge FAILED.\n"
                f"Criteria: {judge_spec['criteria']}\n"
                f"Reasoning: {reasoning}\n"
                f"Response: {response}"
            )

    # ------------------------------------------------------------------
    # 9. LangSmith feedback scores (always reached)
    # ------------------------------------------------------------------
    all_failures = routing_failures + content_failures
    failure_reason = all_failures[0] if all_failures else None

    t.log_outputs({"failure_reason": failure_reason})
    t.log_feedback(key="tool_routing", score=0.0 if routing_failures else 1.0)
    t.log_feedback(key="response_content", score=0.0 if content_failures else 1.0)
    if judge_passed is not None:
        t.log_feedback(key="llm_judge", score=1.0 if judge_passed else 0.0)

    if all_failures:
        pytest.fail(all_failures[0])
