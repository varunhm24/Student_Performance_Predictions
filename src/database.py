import sqlite3
import os

# Database location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "student_performance.db")


def get_connection():
    """Create and return a database connection."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the predictions table if it does not exist."""
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_marks REAL NOT NULL,
            assignments REAL NOT NULL,
            internal_marks REAL NOT NULL,
            prediction TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully!")
