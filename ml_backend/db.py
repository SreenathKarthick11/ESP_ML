import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "keys.db")

def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS api_keys (key TEXT PRIMARY KEY)')
        # Optionally insert a test key:
        # c.execute('INSERT INTO api_keys (key) VALUES (?)', ("your-test-key",))
        conn.commit()
        conn.close()
