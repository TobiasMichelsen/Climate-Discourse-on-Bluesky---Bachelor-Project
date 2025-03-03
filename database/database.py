import sqlite3
from firehose.interested_records import get_interested_records
import json

DB_PATH = "firehose.db"

def setup_database():
    """Creates tables dynamically based on interested records."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    INTERESTED_RECORDS = get_interested_records()

    for event_type in INTERESTED_RECORDS.keys():
        table_name = event_type.replace(".", "_")  # Convert event type to a valid SQL table name

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                repo TEXT,
                data TEXT
            )
        """)

    conn.commit()
    conn.close()
    print("✅ Database setup completed.")

def get_connection():
    """Returns a database connection."""
    return sqlite3.connect(DB_PATH)
