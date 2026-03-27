# Why Evals?

## The problem

This bot wraps an LLM agent over a live database. When you change anything — the system prompt, a tool description, the model, retrieval logic — you have no simple way to know whether the agent still behaves correctly. Unit tests on individual functions won't catch it: the failure mode is the *agent's reasoning*, not a code bug. You need to run the agent end-to-end against representative inputs and assert on its outputs.

## Why not just run it manually?

Manual smoke-testing is slow, inconsistent, and easy to skip under time pressure. It also produces no artifact — you can't tell whether a change regressed something you tested last week. Evals solve both problems: they're fast enough to run on every meaningful change, and they leave a record you can diff across runs.

## Why pytest?

The agent already lives in Python. Pytest gives us parameterization, fixtures, caching, and a familiar CLI with no extra infrastructure. Each test case is a JSON object; the framework loads them and runs the agent — no bespoke eval harness to maintain.

## Suite structure

22 test cases across four categories, each targeting a distinct failure mode:

| Category | Cases | What it guards against |
|---|---|---|
| `happy_path` | 5 | Core regressions: wrong tool routed, bad args, missing data in response |
| `clarification` | 7 | Agent acting without required info (e.g. querying `get_customer_info` without an ID) |
| `edge` | 5 | Silent failures: unknown artist, invalid ID, out-of-scope requests |
| `adversarial` | 5 | Prompt injection, SQL injection input, hallucinated data |

### Three check layers per case

Every case can independently assert at three levels:

1. **Routing** (deterministic) — which tools were called, in what order, with what arguments
2. **Content** (deterministic) — required and forbidden keywords in the final response
3. **LLM judge** (probabilistic) — a GPT-4o-mini judge evaluates natural-language criteria when keyword matching is too coarse

Routing and content checks are fast and free. The LLM judge is opt-in and catches semantic failures that keywords miss.

## The smoke suite

9 of the 22 cases are tagged `smoke: true` — one representative case per failure mode. Running `pytest evals/ --smoke` covers all four categories in under a minute (with cache). This is the intended workflow when iterating on the agent:

1. Change the system prompt, tool description, or model.
2. Run `pytest evals/ --smoke --clear-cache` to force fresh API calls.
3. Check the summary table and delta from the previous run.
4. If smoke passes, run the full suite before merging.

## Caching

LLM responses are cached in a local SQLite file (`.langchain.db`), keyed on the full prompt and model config. Reruns are nearly instant and cost nothing. The cache invalidates automatically when the system prompt, model, or tool definitions change. This makes it practical to run evals frequently without worrying about API cost.

## LangSmith integration

If `LANGSMITH_API_KEY` is set, each run is recorded as a named experiment with per-test scores. This creates a persistent history across agent changes — useful for tracking whether a prompt improvement actually moved the needle, or which category degraded after a model upgrade.

## Design principle

New bugs found in production should become new test cases. The suite is only as useful as it is complete, and real failure modes are the best source of new cases.
