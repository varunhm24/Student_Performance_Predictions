import sqlite3
import os
import pandas as pd

database = "student_predictions.db"

if not os.path.exists(database):
    print("Database file not found!")
else:
    connection = sqlite3.connect(database)

    df = pd.read_sql_query(
        "SELECT * FROM predictions",
        connection
    )

    print("\n========== SAVED PREDICTIONS ==========")
    print(df.to_string(index=False))

    print("\nTotal predictions:", len(df))

    connection.close()