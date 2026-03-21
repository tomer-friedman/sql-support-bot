"""
Tool routing tests.

Deterministic checks that the agent calls the correct tool(s) for each
query type, and never calls forbidden tools. No LLM judge required —
we inspect the AIMessage.tool_calls in the returned message list.
"""

import pytest
from evals.helpers import get_tool_names, invoke_and_get_run_id, log_eval_feedback

ALL_TOOLS = [
    "get_albums_by_artist",
    "get_tracks_by_artist",
    "check_for_songs",
    "get_customer_info",
]

ROUTING_CASES = [
    pytest.param(
        "album_search_acdc",
        [{"role": "user", "content": "What albums does AC/DC have?"}],
        ["get_albums_by_artist"],
        [],
        id="album_search_acdc",
    ),
    pytest.param(
        "track_search_metallica",
        [{"role": "user", "content": "What songs does Metallica have?"}],
        ["get_tracks_by_artist"],
        [],
        id="track_search_metallica",
    ),
    pytest.param(
        "song_title_let_there_be_rock",
        [{"role": "user", "content": "Do you have a song called 'Let There Be Rock'?"}],
        ["check_for_songs"],
        [],
        id="song_title_let_there_be_rock",
    ),
    pytest.param(
        "customer_gating_no_id",
        [{"role": "user", "content": "I need my account info"}],
        [],
        ["get_customer_info"],
        id="customer_gating_no_id",
    ),
    pytest.param(
        "customer_with_id",
        [{"role": "user", "content": "My customer ID is 5, show me my account"}],
        ["get_customer_info"],
        [],
        id="customer_with_id",
    ),
    pytest.param(
        "out_of_scope_weather",
        [{"role": "user", "content": "What's the weather today?"}],
        [],
        ALL_TOOLS,
        id="out_of_scope_weather",
    ),
]


@pytest.mark.parametrize(
    "test_name,messages,expected_tools,forbidden_tools", ROUTING_CASES
)
def test_tool_routing(agent, test_name, messages, expected_tools, forbidden_tools):
    result, run_id = invoke_and_get_run_id(agent, messages)
    called = get_tool_names(result["messages"])

    routing_passed = True
    for tool in expected_tools:
        if tool not in called:
            routing_passed = False
            pytest.fail(f"[{test_name}] Expected tool '{tool}' to be called, got: {called}")

    for tool in forbidden_tools:
        if tool in called:
            routing_passed = False
            pytest.fail(f"[{test_name}] Tool '{tool}' must NOT be called, but was. Called: {called}")

    log_eval_feedback(run_id, test_name, tool_routing_passed=routing_passed)
