"""Shared pytest fixtures for the eval suite."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from agent import create_agent, get_engine

# Load .env at import time so that the @pytest.mark.langsmith plugin
# (which initializes its LangSmith client during pytest startup, before
# any fixtures run) can read LANGSMITH_API_KEY from the environment.
load_dotenv()

# Module-level list populated by the eval_results fixture during each test.
# Read by pytest_terminal_summary to build the summary table.
_eval_results: list[dict] = []

_RESULTS_DIR = Path(__file__).parent / "results"
_LATEST_FULL_JSON = _RESULTS_DIR / "latest.json"
_LATEST_SMOKE_JSON = _RESULTS_DIR / "latest_smoke.json"


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
    parser.addoption(
        "--no-llm-judge",
        action="store_true",
        default=False,
        help="Skip LLM-as-judge; only deterministic routing and content checks run.",
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


@pytest.fixture
def agent(db_engine):
    """Fresh agent per test to prevent any internal state from bleeding
    between test cases. create_agent() only compiles a graph — no API calls —
    so the per-test overhead is negligible."""
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


@pytest.fixture(scope="session")
def eval_results():
    """Shared list; each test appends one result dict.
    Read by pytest_terminal_summary to build the table."""
    return _eval_results


# ---------------------------------------------------------------------------
# Terminal summary — printed once after all tests complete
# ---------------------------------------------------------------------------

def _fmt_frac(passed: int, total: int) -> str:
    if total == 0:
        return "-"
    pct = passed * 100 // total
    return f"{passed}/{total} ({pct:>3}%)"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _eval_results:
        return

    is_smoke = config.getoption("--smoke", default=False)
    _LATEST_JSON = _LATEST_SMOKE_JSON if is_smoke else _LATEST_FULL_JSON

    # ---- load previous run for delta ----
    previous: dict | None = None
    if _LATEST_JSON.exists():
        try:
            with open(_LATEST_JSON) as f:
                previous = json.load(f)
        except Exception:
            pass

    # ---- aggregate by category ----
    categories = sorted({r["category"] for r in _eval_results})
    cat_data: dict[str, dict] = {}
    for cat in categories:
        rows = [r for r in _eval_results if r["category"] == cat]
        cat_data[cat] = {
            "passed":          sum(1 for r in rows if r["overall_passed"]),
            "total":           len(rows),
            "routing_passed":  sum(r["routing_passed"] for r in rows),
            "routing_total":   sum(r["routing_total"] for r in rows),
            "content_passed":  sum(r["content_passed"] for r in rows),
            "content_total":   sum(r["content_total"] for r in rows),
            "judge_passed":    sum(1 for r in rows if r["judge_passed"] is True),
            "judge_total":     sum(1 for r in rows if r["judge_passed"] is not None),
        }

    totals = {k: sum(cat_data[c][k] for c in categories)
              for k in ("passed", "total", "routing_passed", "routing_total",
                        "content_passed", "content_total", "judge_passed", "judge_total")}

    # ---- print table ----
    terminalreporter.write_sep("=", "Eval Summary")

    header = (
        f"{'Category':<20}  {'Tests':>6}  {'Routing':>16}  {'Content':>16}  {'Judge':>14}"
    )
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))

    for cat in categories:
        d = cat_data[cat]
        terminalreporter.write_line(
            f"{cat:<20}  {_fmt_frac(d['passed'], d['total']):>6}  "
            f"{_fmt_frac(d['routing_passed'], d['routing_total']):>16}  "
            f"{_fmt_frac(d['content_passed'], d['content_total']):>16}  "
            f"{_fmt_frac(d['judge_passed'], d['judge_total']):>14}"
        )

    terminalreporter.write_line("-" * len(header))
    terminalreporter.write_line(
        f"{'TOTAL':<20}  {_fmt_frac(totals['passed'], totals['total']):>6}  "
        f"{_fmt_frac(totals['routing_passed'], totals['routing_total']):>16}  "
        f"{_fmt_frac(totals['content_passed'], totals['content_total']):>16}  "
        f"{_fmt_frac(totals['judge_passed'], totals['judge_total']):>14}"
    )

    # ---- delta from previous run ----
    if previous and previous.get("by_category"):
        terminalreporter.write_line("")
        terminalreporter.write_line("Delta from previous run:")
        prev_by_cat = previous["by_category"]
        any_delta = False
        for cat in categories:
            if cat not in prev_by_cat:
                continue
            prev_d = prev_by_cat[cat]
            curr_d = cat_data[cat]
            prev_pct = prev_d["passed"] * 100 // max(prev_d["total"], 1)
            curr_pct = curr_d["passed"] * 100 // max(curr_d["total"], 1)
            delta = curr_pct - prev_pct
            if delta != 0:
                any_delta = True
                sign = "+" if delta > 0 else ""
                arrow = "✓" if delta > 0 else "✗"
                terminalreporter.write_line(
                    f"  {cat:<20}  {prev_pct:>3}% → {curr_pct:>3}%  {sign}{delta}%  {arrow}"
                )

        # overall delta
        if totals["total"] > 0 and previous.get("total"):
            prev_total = previous["total"]
            prev_pct = prev_total["passed"] * 100 // max(prev_total["total"], 1)
            curr_pct = totals["passed"] * 100 // max(totals["total"], 1)
            delta = curr_pct - prev_pct
            if delta != 0:
                any_delta = True
                sign = "+" if delta > 0 else ""
                arrow = "✓" if delta > 0 else "✗"
                terminalreporter.write_line(
                    f"  {'TOTAL':<20}  {prev_pct:>3}% → {curr_pct:>3}%  {sign}{delta}%  {arrow}"
                )

        if not any_delta:
            terminalreporter.write_line("  No changes from previous run.")

    terminalreporter.write_sep("=", "")

    # ---- save current run to results/latest.json ----
    _RESULTS_DIR.mkdir(exist_ok=True)
    current = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": {"passed": totals["passed"], "total": totals["total"]},
        "by_category": {
            cat: {"passed": d["passed"], "total": d["total"]}
            for cat, d in cat_data.items()
        },
        "cases": [
            {
                "name": r["name"],
                "category": r["category"],
                "smoke": r["smoke"],
                "passed": r["overall_passed"],
            }
            for r in _eval_results
        ],
    }
    with open(_LATEST_JSON, "w") as f:
        json.dump(current, f, indent=2)
