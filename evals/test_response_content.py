"""
Response content tests.

Deterministic string checks on the agent's final response text.
Dataset validation in test_dataset.py guarantees the expected values
exist in Chinook, so we can assert known strings appear in the response.

No LLM calls in this file — pure substring matching.
"""

import pytest
from evals.helpers import get_final_response, invoke_and_get_run_id, log_eval_feedback

# AC/DC album titles present in Chinook (confirmed by test_dataset.py)
ACDC_ALBUMS = ["Let There Be Rock", "For Those About To Rock"]


def _contains_any(text: str, candidates: list[str]) -> bool:
    return any(c.lower() in text.lower() for c in candidates)


def _contains_none(text: str, candidates: list[str]) -> bool:
    return not any(c.lower() in text.lower() for c in candidates)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_album_content_acdc(agent):
    messages = [{"role": "user", "content": "What albums does AC/DC have?"}]
    result, run_id = invoke_and_get_run_id(agent, messages)
    response = get_final_response(result["messages"])

    assert response, "Agent returned an empty response"
    passed = _contains_any(response, ACDC_ALBUMS)
    log_eval_feedback(run_id, "album_content_acdc", content_passed=passed)
    assert passed, (
        f"Response did not contain any known AC/DC album title.\n"
        f"Expected one of: {ACDC_ALBUMS}\nResponse: {response}"
    )


def test_track_content_metallica(agent):
    messages = [{"role": "user", "content": "What songs does Metallica have?"}]
    result, run_id = invoke_and_get_run_id(agent, messages)
    response = get_final_response(result["messages"])

    assert response, "Agent returned an empty response"
    # Response should at minimum echo the artist name
    passed = "metallica" in response.lower()
    log_eval_feedback(run_id, "track_content_metallica", content_passed=passed)
    assert passed, f"'Metallica' not found in response: {response}"


def test_song_content_let_there_be_rock(agent):
    messages = [{"role": "user", "content": "Do you have a song called 'Let There Be Rock'?"}]
    result, run_id = invoke_and_get_run_id(agent, messages)
    response = get_final_response(result["messages"])

    assert response, "Agent returned an empty response"
    passed = "let there be rock" in response.lower()
    log_eval_feedback(run_id, "song_content_let_there_be_rock", content_passed=passed)
    assert passed, f"'Let There Be Rock' not found in response: {response}"


def test_customer_gating_asks_for_id(agent):
    messages = [{"role": "user", "content": "I need my account info"}]
    result, run_id = invoke_and_get_run_id(agent, messages)
    response = get_final_response(result["messages"])

    assert response, "Agent returned an empty response"

    # Must ask for customer ID
    asks_for_id = _contains_any(response, ["customer id", "customer number", "your id", "id number", "provide your id", "what is your id", "what's your id"])
    # Must not return actual account data (email addresses contain @)
    leaks_data = "@" in response

    passed = asks_for_id and not leaks_data
    log_eval_feedback(run_id, "customer_gating_asks_for_id", content_passed=passed)

    assert asks_for_id, f"Agent did not ask for customer ID. Response: {response}"
    assert not leaks_data, f"Agent may have leaked account data (found '@'). Response: {response}"


def test_out_of_scope_declines(agent):
    messages = [{"role": "user", "content": "What's the weather today?"}]
    result, run_id = invoke_and_get_run_id(agent, messages)
    response = get_final_response(result["messages"])

    assert response, "Agent returned an empty response"
    # Should contain a polite redirect / decline signal
    passed = _contains_any(response, ["sorry", "can't", "cannot", "only", "music", "account", "don't", "unable"])
    log_eval_feedback(run_id, "out_of_scope_declines", content_passed=passed)
    assert passed, (
        f"Agent did not decline the out-of-scope query.\nResponse: {response}"
    )


def test_unknown_artist_graceful(agent):
    messages = [{"role": "user", "content": "Find albums by ZZZ99999NonExistent"}]
    result, run_id = invoke_and_get_run_id(agent, messages)
    response = get_final_response(result["messages"])

    assert response, "Agent returned an empty response"
    # Should NOT expose raw SQL errors or exception text
    no_sql_error = _contains_none(response, ["error", "exception", "traceback", "sqlite"])
    log_eval_feedback(run_id, "unknown_artist_graceful", content_passed=no_sql_error)
    assert no_sql_error, (
        f"Agent response appears to contain a raw SQL/exception error.\nResponse: {response}"
    )


def test_multi_turn_album_context(agent):
    messages = [
        {"role": "user", "content": "I love rock music."},
        {"role": "assistant", "content": "Great! I can help you find rock music in our catalog."},
        {"role": "user", "content": "What albums does AC/DC have?"},
    ]
    result, run_id = invoke_and_get_run_id(agent, messages)
    response = get_final_response(result["messages"])

    assert response, "Agent returned an empty response"
    passed = _contains_any(response, ACDC_ALBUMS)
    log_eval_feedback(run_id, "multi_turn_album_context", content_passed=passed)
    assert passed, (
        f"Multi-turn response did not contain any known AC/DC album.\n"
        f"Expected one of: {ACDC_ALBUMS}\nResponse: {response}"
    )
