"""
LangSmith eval report.

Prints a pass-rate summary for the most recent eval runs logged to LangSmith.

Usage:
    uv run python evals/report.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("LANGCHAIN_API_KEY not set — no LangSmith data to report.")
        print("Run: uv run pytest evals/ -v  to execute the eval suite.")
        sys.exit(0)

    from langsmith import Client

    client = Client()
    project = os.getenv("LANGCHAIN_PROJECT", "sql-support-bot-evals")

    # Collect feedback from the last 7 days
    since = datetime.now(timezone.utc) - timedelta(days=7)

    try:
        runs = list(client.list_runs(
            project_name=project,
            run_type="chain",
            start_time=since,
        ))
    except Exception as e:
        print(f"Could not fetch runs from LangSmith: {e}")
        sys.exit(1)

    if not runs:
        print(f"No runs found in LangSmith project '{project}' in the last 7 days.")
        print("Run: uv run pytest evals/ -v  to generate eval data.")
        sys.exit(0)

    run_ids = [str(r.id) for r in runs]

    routing_scores = []
    content_scores = []

    for run_id in run_ids:
        feedbacks = list(client.list_feedback(run_ids=[run_id]))
        for fb in feedbacks:
            if fb.key == "tool_routing" and fb.score is not None:
                routing_scores.append(fb.score)
            elif fb.key == "response_content" and fb.score is not None:
                content_scores.append(fb.score)

    def rate(scores: list[float]) -> str:
        if not scores:
            return "N/A"
        passed = sum(1 for s in scores if s >= 1.0)
        pct = passed / len(scores) * 100
        return f"{pct:.0f}%  ({passed}/{len(scores)})"

    all_scores = routing_scores + content_scores

    print()
    print("=" * 52)
    print("  SQL Support Bot — Eval Report (last 7 days)")
    print(f"  Project: {project}")
    print("=" * 52)
    print(f"  Tool Routing      : {rate(routing_scores)}")
    print(f"  Response Content  : {rate(content_scores)}")
    print(f"  Overall           : {rate(all_scores)}")
    print("=" * 52)
    print()


if __name__ == "__main__":
    main()
