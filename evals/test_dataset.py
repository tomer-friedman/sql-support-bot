"""
Dataset validation tests.

These tests query the Chinook DB directly (no agent involved) to confirm
that all the data our other eval tests depend on actually exists.
By default they are skipped; pass --run-db-tests to run them.
"""

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.db_validation


def test_acdc_exists(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT Name FROM Artist WHERE Name LIKE '%AC/DC%'")
        ).fetchall()
    assert rows, "AC/DC not found in Artist table"


def test_acdc_has_albums(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT Album.Title FROM Album
                JOIN Artist ON Album.ArtistId = Artist.ArtistId
                WHERE Artist.Name LIKE '%AC/DC%'
            """)
        ).fetchall()
    titles = [r[0] for r in rows]
    assert titles, "AC/DC has no albums in the database"
    # Matches required_keywords in album_search_acdc and multi_turn_album_after_greeting
    for needle in (
        "Let There Be Rock",
        "For Those About To Rock We Salute You",
    ):
        assert any(needle in t for t in titles), (
            f"Expected album title containing {needle!r} for AC/DC, got: {titles}"
        )


def test_metallica_exists(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT Name FROM Artist WHERE Name LIKE '%Metallica%'")
        ).fetchall()
    assert rows, "Metallica not found in Artist table"


def test_metallica_has_tracks(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT Track.Name FROM Track
                JOIN Album ON Track.AlbumId = Album.AlbumId
                JOIN Artist ON Album.ArtistId = Artist.ArtistId
                WHERE Artist.Name LIKE '%Metallica%'
            """)
        ).fetchall()
    assert rows, "Metallica has no tracks in the database"


def test_let_there_be_rock_track(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT Name FROM Track WHERE Name LIKE '%Let There Be Rock%'")
        ).fetchall()
    assert rows, "'Let There Be Rock' track not found in Track table"


def test_customer_5_exists(db_engine):
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT CustomerId FROM Customer WHERE CustomerId = 5")
        ).fetchall()
    assert rows, "Customer with ID 5 not found in Customer table"


def test_customer_5_firstname_matches_eval_anchor(db_engine):
    """Required keyword in customer_lookup_with_id and related evals."""
    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT FirstName FROM Customer WHERE CustomerId = 5")
        ).fetchone()
    assert row and row[0] == "František", (
        f"Expected FirstName 'František' for CustomerId 5, got {row!r}"
    )


def test_customer_7_exists(db_engine):
    """clarification_account_then_provides_id expects a successful lookup for ID 7."""
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT CustomerId FROM Customer WHERE CustomerId = 7")
        ).fetchall()
    assert rows, "Customer with ID 7 not found in Customer table"


def test_customer_9999_absent(db_engine):
    """edge_nonexistent_customer_id assumes this ID does not exist."""
    with db_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT CustomerId FROM Customer WHERE CustomerId = 9999")
        ).fetchall()
    assert not rows, "Customer 9999 should not exist for edge_nonexistent_customer_id eval"


def test_customer_count(db_engine):
    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM Customer")).scalar()
    assert 50 <= count <= 70, (
        f"Unexpected customer count {count} — expected between 50 and 70. "
        "The dataset may have changed significantly."
    )
