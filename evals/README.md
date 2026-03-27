# Eval Suite

Pytest-based evaluation suite for the SQL support bot. 22 test cases across four categories, with a 9-case smoke subset for fast iteration.

## Quick start

```bash
# Fast: smoke suite across all categories (~9 cases, <1 min with cache)
pytest evals/ --smoke

# Full suite (22 cases)
pytest evals/

# Deterministic only — no LLM judge, no API cost beyond the agent itself
pytest evals/ --smoke --no-llm-judge

# Force fresh API calls (e.g. after changing the system prompt)
pytest evals/ --smoke --clear-cache
```

## Test categories

| Category | Cases | What it covers |
|---|---|---|
| `happy_path` | 5 | Core tool invocations with known-good inputs: album search, track search, song title lookup, account lookup with ID, multi-turn context |
| `clarification` | 7 | Agent asks for missing info before acting — most critically, requesting a customer ID before calling `get_customer_info` |
| `edge` | 5 | Graceful failure: unknown artist, invalid customer ID, out-of-scope requests, too many results, special characters |
| `adversarial` | 5 | Prompt injection, SQL injection input, gibberish, refusal to hallucinate customer data, refusal to hallucinate music catalog data |

## Smoke coverage

The 9 smoke cases (tagged `smoke: true` in `test_cases.json`) are chosen to cover one representative scenario per failure mode. Each smoke case has a `smoke_rationale` field explaining why it was selected. Together they give a cross-category signal in under a minute.

## Check layers

Each test case can define up to three layers of checks:

1. **Routing** (deterministic) — which tools were called, in what order, and with what arguments
2. **Content** (deterministic) — final response must be non-empty; optional required and forbidden keywords
3. **LLM judge** (probabilistic, opt-out) — natural language criteria evaluated by `gpt-4o-mini`

All three layers produce independent scores reported in the terminal summary.

## Flags

| Flag | Effect |
|---|---|
| `--smoke` | Run only the 9 smoke-tagged cases |
| `--no-llm-judge` | Skip LLM-as-judge; only deterministic checks run |
| `--no-tool-checks` | Skip tool routing checks; only content and judge checks run |
| `--no-cache` | Disable SQLite response cache; every call hits the API |
| `--clear-cache` | Delete the cache before the session, then rebuild it |
| `--run-db-tests` | Run dataset validation tests against the local Chinook DB (skipped by default) |

## Caching

LLM responses are cached in `.langchain.db` (SQLite, keyed on prompt + model config). This makes reruns nearly instant and cost-free. The cache invalidates automatically when the system prompt, model, or tool definitions change — or you can force a rebuild with `--clear-cache`.

## LangSmith

If `LANGSMITH_API_KEY` is set, each run is recorded as a named experiment (`sql-evals-YYYYMMDD-HHMMSS-{smoke|full}`) with per-test scores for `tool_routing`, `response_content`, and `llm_judge`. This creates a persistent history you can diff across agent changes.

## Adding cases

All test cases live in `test_cases.json`. Each case is a JSON object with:

- `name`, `category`, `smoke` (bool)
- `smoke_rationale` — required if `smoke: true`; explains why this case represents its category in the smoke suite
- `input` — string (single-turn) or list of `{role, content}` dicts (multi-turn)
- `expected_tool_calls`, `forbidden_tools`, `check_order`, `max_tool_calls`
- `required_keywords`, `forbidden_keywords`
- `llm_judge` — `{criteria, scoring}` or `null`

New bugs found in production should become new test cases here.
