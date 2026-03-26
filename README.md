# SQL Support Bot

Customer support chatbot built with DeepAgents that interacts with a SQL database (Chinook music store) to answer questions about music and customer accounts.

## What This Bot Does

This bot can help customers:
1. **Find music** - search for songs, albums, and artists in the catalog
2. **Access account info** - look up customer account details

The bot uses DeepAgents to autonomously decide which tools to use based on the customer's query.

## Setup

Install dependencies:
```bash
uv sync
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Python Script
```bash
uv run python agent.py
```

Type your questions and the agent will respond. Type `quit` to exit.

### Jupyter Notebook
```bash
uv run jupyter notebook
```

Then open `agent.ipynb` and run cells sequentially to interact with the agent.

## Example Queries

- "Can you help me find songs by The Beatles?"
- "What albums does Pink Floyd have?"
- "What's the email for customer ID 5?"

## Exploring the database manually

The app loads Chinook into **memory**; for a persistent file you can open in the `sqlite3` shell, [DB Browser for SQLite](https://sqlitebrowser.org/), or any SQLite client:

```bash
./scripts/build_chinook_db.sh
sqlite3 data/chinook.db
```

Inside `sqlite3`, try `.tables`, `.schema Album`, and your `SELECT` statements. Re-run the script anytime to refresh `data/chinook.db` from the upstream SQL.

## How It Works

- **Database**: Uses the Chinook database (downloads automatically on first run)
- **Tools**: Agent has access to 4 tools for searching music and looking up customer info
- **Routing**: DeepAgents automatically decides which tool(s) to use based on the query
