"""
Dataset validation tests.

These tests query the Chinook DB directly (no agent involved) to confirm
that all the data our other eval tests depend on actually exists.
Run these first — if they fail the rest of the suite is unreliable.
"""

from sqlalchemy import text


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
    assert any("Let There Be Rock" in t for t in titles), (
        f"Expected 'Let There Be Rock' among AC/DC albums, got: {titles}"
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


def test_customer_count(db_engine):
    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM Customer")).scalar()
    assert 50 <= count <= 70, (
        f"Unexpected customer count {count} — expected between 50 and 70. "
        "The dataset may have changed significantly."
    )
