#!/usr/bin/env bash
# Build data/chinook.db from the same Chinook SQL script used by agent.py (for manual CLI/GUI exploration).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
OUT="${ROOT}/data/chinook.db"

mkdir -p "${ROOT}/data"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

curl -fsSL -o "$TMP" "$URL"
rm -f "$OUT"
sqlite3 "$OUT" < "$TMP"
echo "Wrote ${OUT}"
