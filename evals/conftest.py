"""Shared pytest fixtures for the eval suite."""

import os
import pytest
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from agent import create_agent, get_engine

# Load .env at import time so that the @pytest.mark.langsmith plugin
# (which initializes its LangSmith client during pytest startup, before
# any fixtures run) can read LANGSMITH_API_KEY from the environment.
load_dotenv()


def pytest_addoption(parser):
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="Run only the smoke-tagged subset of test cases.",
    )
    parser.addoption(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable the LLM response cache; every invocation hits the API.",
    )
    parser.addoption(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Delete the LLM response cache before the session, then rebuild it.",
    )


@pytest.fixture(scope="session", autouse=True)
def setup_llm_cache(pytestconfig):
    """Cache LLM responses to disk so reruns skip API calls entirely.
    Cache key = full prompt + LLM config, so it invalidates automatically
    when the agent's system prompt, model, or tools change.

    --no-cache    skip the cache entirely (every run hits the API)
    --clear-cache delete the existing cache file before the session
    """
    cache_path = os.path.join(os.path.dirname(__file__), ".langchain.db")

    if pytestconfig.getoption("--clear-cache") and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"\n[cache] cleared {cache_path}")

    if pytestconfig.getoption("--no-cache"):
        return  # leave LangChain cache unset — all calls go to the API

    set_llm_cache(SQLiteCache(database_path=cache_path))


@pytest.fixture(scope="session")
def db_engine():
    """Initialise the lazy DB singleton and return the engine.
    Used by test_dataset.py for direct SQL queries.
    Calling get_engine() here means the agent's tools will reuse this
    connection rather than triggering a second Chinook download."""
    return get_engine()


@pytest.fixture(scope="session")
def agent(db_engine):
    """Create the agent once per session. db_engine fixture runs first,
    so the lazy singleton is already initialised when create_agent() runs."""
    return create_agent()


@pytest.fixture(scope="session")
def llm_judge_client():
    """OpenEvals LLM-as-judge factory. Single shared instance across all cases;
    per-case criteria are passed as a prompt template variable at call time."""
    from openevals.llm import create_llm_as_judge

    prompt = """\
You are an impartial evaluator of an AI assistant that helps users with \
a music catalog and customer accounts.

USER INPUT:
{user_input}

AGENT RESPONSE:
{agent_response}

EVALUATION CRITERIA:
{criteria}

Score 1 if the agent response satisfies the criteria, 0 if it does not. \
Explain your reasoning briefly."""

    return create_llm_as_judge(
        prompt=prompt,
        model="openai:gpt-4o-mini",
    )
