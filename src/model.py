
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "data", "student_performance.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "student_performance_model.pkl"
)

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments",
    "Internal_Marks"
]


def train_model():
    """Train and save the student performance prediction model."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    required_columns = FEATURES + ["Final_Result"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    data = data[required_columns].dropna()

    data["Final_Result"] = (
        data["Final_Result"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    valid_results = {"Pass", "Fail"}

    if not set(data["Final_Result"].unique()).issubset(valid_results):
        raise ValueError(
            "Final_Result must contain only Pass or Fail."
        )

    X = data[FEATURES]
    y = data["Final_Result"].map({
        "Fail": 0,
        "Pass": 1
    })

    if y.nunique() != 2:
        raise ValueError(
            "Dataset must contain both Pass and Fail records."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    print("\nMODEL TRAINING COMPLETED")
    print("------------------------")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Fail", "Pass"],
            zero_division=0
        )
    )

    return model


def load_model():
    """Load the saved model or train it if unavailable."""

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    return train_model()


def predict_student(input_data):
    """Predict whether a student will Pass or Fail."""

    model = load_model()

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    result = model.predict(input_df)[0]

    return "Pass" if int(result) == 1 else "Fail"


if __name__ == "__main__":
    train_model()
