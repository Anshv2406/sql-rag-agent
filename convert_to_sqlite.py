import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData

load_dotenv()

PG_URL = os.getenv("DATABASE_URL")  # your existing Postgres URL
SQLITE_PATH = "chinook.db"
SQLITE_URL = f"sqlite:///{SQLITE_PATH}"

print(f"Reading from: {PG_URL.split('@')[-1]}")
pg_engine = create_engine(PG_URL)

print(f"Writing to: {SQLITE_PATH}")
sqlite_engine = create_engine(SQLITE_URL)

metadata = MetaData()
metadata.reflect(bind=pg_engine)

metadata.create_all(sqlite_engine)

with pg_engine.connect() as pg_conn, sqlite_engine.connect() as sqlite_conn:
    for table in metadata.sorted_tables:
        print(f"Copying table: {table.name}")
        rows = pg_conn.execute(table.select()).fetchall()
        if rows:
            sqlite_conn.execute(table.insert(), [dict(row._mapping) for row in rows])
        sqlite_conn.commit()
        print(f"  -> {len(rows)} rows copied")

print(f"\nDone. SQLite file created at: {SQLITE_PATH}")