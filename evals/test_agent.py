"""
Unified agent eval test.

Each case in test_cases.json is tested once, covering:
  1. Deterministic tool routing checks (expected calls, arg values, forbidden tools, order, count)
  2. Deterministic content checks (required/forbidden keywords)
  3. LLM-as-judge for subjective criteria (if "llm_judge" is set in the case)

All results are logged to LangSmith when LANGCHAIN_API_KEY is set.
"""

import json
import pytest
from pathlib import Path

from evals.helpers import (
    get_tool_names,
    get_tool_call_args,
    get_final_response,
    invoke_and_get_run_id,
    run_llm_judge,
    log_eval_feedback,
)

_CASES_PATH = Path(__file__).parent / "test_cases.json"


def _load_cases():
    with open(_CASES_PATH) as f:
        return json.load(f)


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = _load_cases()
        metafunc.parametrize("case", cases, ids=[c["name"] for c in cases])


def test_agent(agent, llm_judge_client, case):
    # --- Build message list ---
    inp = case["input"]
    if isinstance(inp, str):
        messages = [{"role": "user", "content": inp}]
    else:
        messages = inp  # already a list of {role, content} dicts

    result, run_id = invoke_and_get_run_id(agent, messages)
    all_messages = result["messages"]
    called_tools = get_tool_names(all_messages)
    response = get_final_response(all_messages)

    routing_passed = True
    content_passed = True
    judge_passed = None  # None means "not evaluated"

    # ------------------------------------------------------------------
    # 1. Expected tools called (and arg substring checks)
    # ------------------------------------------------------------------
    for expected in case.get("expected_tool_calls", []):
        tool_name = expected["tool"]
        if tool_name not in called_tools:
            routing_passed = False
            pytest.fail(
                f"[{case['name']}] Expected tool '{tool_name}' was not called. "
                f"Actual calls: {called_tools}"
            )

        if "args" in expected:
            actual_args = get_tool_call_args(all_messages, tool_name)
            for arg_key, expected_substr in expected["args"].items():
                actual_val = str(actual_args.get(arg_key, ""))
                if expected_substr.lower() not in actual_val.lower():
                    routing_passed = False
                    pytest.fail(
                        f"[{case['name']}] Tool '{tool_name}' arg '{arg_key}': "
                        f"expected substring '{expected_substr}' not found in '{actual_val}'"
                    )

    # ------------------------------------------------------------------
    # 2. Forbidden tools
    # ------------------------------------------------------------------
    for tool_name in case.get("forbidden_tools", []):
        if tool_name in called_tools:
            routing_passed = False
            pytest.fail(
                f"[{case['name']}] Forbidden tool '{tool_name}' was called. "
                f"Actual calls: {called_tools}"
            )

    # ------------------------------------------------------------------
    # 3. Tool call ordering
    # ------------------------------------------------------------------
    if case.get("check_order") and len(case.get("expected_tool_calls", [])) > 1:
        expected_sequence = [e["tool"] for e in case["expected_tool_calls"]]
        filtered = [t for t in called_tools if t in expected_sequence]
        if filtered != expected_sequence:
            routing_passed = False
            pytest.fail(
                f"[{case['name']}] Tool call order mismatch. "
                f"Expected: {expected_sequence}, got: {filtered}"
            )

    # ------------------------------------------------------------------
    # 4. Max tool calls
    # ------------------------------------------------------------------
    max_calls = case.get("max_tool_calls")
    if max_calls is not None and len(called_tools) > max_calls:
        routing_passed = False
        pytest.fail(
            f"[{case['name']}] Too many tool calls: {len(called_tools)} > {max_calls}. "
            f"Calls: {called_tools}"
        )

    # ------------------------------------------------------------------
    # 5. Non-empty response
    # ------------------------------------------------------------------
    if case.get("response_must_be_nonempty", True) and not response:
        content_passed = False
        pytest.fail(f"[{case['name']}] Agent returned an empty response.")

    # ------------------------------------------------------------------
    # 6. Required keywords
    # ------------------------------------------------------------------
    for kw in case.get("required_keywords", []):
        if kw.lower() not in response.lower():
            content_passed = False
            pytest.fail(
                f"[{case['name']}] Required keyword '{kw}' not found in response.\n"
                f"Response: {response}"
            )

    # ------------------------------------------------------------------
    # 7. Forbidden keywords
    # ------------------------------------------------------------------
    for kw in case.get("forbidden_keywords", []):
        if kw.lower() in response.lower():
            content_passed = False
            pytest.fail(
                f"[{case['name']}] Forbidden keyword '{kw}' found in response.\n"
                f"Response: {response}"
            )

    # ------------------------------------------------------------------
    # 8. LLM-as-judge (subjective checks)
    # ------------------------------------------------------------------
    judge_spec = case.get("llm_judge")
    if judge_spec:
        user_input_text = inp if isinstance(inp, str) else inp[-1]["content"]
        judge_passed = run_llm_judge(
            llm_judge_client=llm_judge_client,
            criteria=judge_spec["criteria"],
            user_input=user_input_text,
            agent_response=response,
            test_name=case["name"],
        )
        if not judge_passed:
            pytest.fail(
                f"[{case['name']}] LLM judge FAILED.\n"
                f"Criteria: {judge_spec['criteria']}\n"
                f"Response: {response}"
            )

    # ------------------------------------------------------------------
    # 9. LangSmith feedback
    # ------------------------------------------------------------------
    log_eval_feedback(
        run_id=run_id,
        test_name=case["name"],
        tool_routing_passed=routing_passed,
        content_passed=content_passed,
        judge_passed=judge_passed,
    )
