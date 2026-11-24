
from flask import Flask, render_template, request, redirect, url_for, Response
import joblib
import sqlite3
import csv
import io
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"
DATABASE_PATH = BASE_DIR / "student_predictions.db"

# Load trained machine learning model
model = joblib.load(MODEL_PATH)


# Connect to SQLite database
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# Create database and safely update an existing table
def create_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            study_hours REAL,
            attendance REAL,
            previous_marks REAL,
            assignments REAL,
            internal_marks REAL,
            prediction TEXT,
            created_at TEXT
        )
    """)

    # Add Student_ID to an older database if it does not exist
    columns = [
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    ]

    if "student_id" not in columns:
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN student_id TEXT"
        )

    connection.commit()
    connection.close()


create_database()


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        student_id = request.form["student_id"].strip()

        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])
        assignments = float(request.form["assignments"])
        internal_marks = float(request.form["internal_marks"])

        # Prepare input in the same feature order used during training
        input_data = [[
            study_hours,
            attendance,
            previous_marks,
            assignments,
            internal_marks
        ]]

        result = model.predict(input_data)[0]

        # Convert model output into Pass or Fail
        prediction = "Pass" if int(result) == 1 else "Fail"

        connection = get_connection()

        connection.execute("""
            INSERT INTO predictions (
                student_id,
                study_hours,
                attendance,
                previous_marks,
                assignments,
                internal_marks,
                prediction,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            study_hours,
            attendance,
            previous_marks,
            assignments,
            internal_marks,
            prediction,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        connection.commit()
        connection.close()

        message = (
            f"Prediction completed for Student ID: {student_id}"
        )

        return render_template(
            "index.html",
            prediction=prediction,
            message=message
        )

    return render_template("index.html")


@app.route("/history")
def history():
    connection = get_connection()

    predictions = connection.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


@app.route("/dashboard")
def dashboard():
    connection = get_connection()

    total = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    passed = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 'Pass'"
    ).fetchone()[0]

    failed = connection.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 'Fail'"
    ).fetchone()[0]

    pass_percentage = round((passed / total) * 100, 2) if total else 0
    fail_percentage = round((failed / total) * 100, 2) if total else 0

    recent_predictions = connection.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT 5
    """).fetchall()

    connection.close()

    return render_template(
        "dashboard.html",
        total=total,
        passed=passed,
        failed=failed,
        pass_percentage=pass_percentage,
        fail_percentage=fail_percentage,
        recent_predictions=recent_predictions
    )


@app.route("/export")
def export():
    connection = get_connection()

    records = connection.execute("""
        SELECT
            student_id,
            study_hours,
            attendance,
            previous_marks,
            assignments,
            internal_marks,
            prediction,
            created_at
        FROM predictions
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Student_ID",
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignments",
        "Internal_Marks",
        "Final_Result",
        "Created_At"
    ])

    for record in records:
        writer.writerow(list(record))

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=student_predictions.csv"
        }
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )



