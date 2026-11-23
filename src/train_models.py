
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# STEP 1: LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("dataset/student_performance_cleaned.csv")

print("Dataset loaded successfully!")

# ==========================================
# STEP 2: CHECK DATASET AND SELECT FEATURES
# ==========================================

required_columns = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments",
    "Internal_Marks",
    "Final_Result"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignments",
        "Internal_Marks"
    ]
].copy()

# Convert all supported target formats into 0 and 1
target_values = (
    df["Final_Result"]
    .astype(str)
    .str.strip()
    .str.lower()
)

y = target_values.map({
    "fail": 0,
    "pass": 1,
    "0": 0,
    "1": 1,
    "0.0": 0,
    "1.0": 1
})

# Stop if unexpected target values are present
if y.isna().any():
    print("\nUnexpected Final_Result values:")
    print(df.loc[y.isna(), "Final_Result"].unique())
    raise ValueError("Invalid values found in Final_Result.")

y = y.astype(int)

# Check missing input values
if X.isna().any().any():
    raise ValueError("Input features contain missing values.")

# Check input data is numeric
X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError("Input features contain non-numeric values.")

print("\nTarget encoding:")
print(y.value_counts().rename(index={0: "Fail", 1: "Pass"}))

print("\nInput feature shape:", X.shape)
print("Target variable shape:", y.shape)

# ==========================================
# STEP 3: SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLITTING ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# STEP 4: CREATE DIRECTORIES
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==========================================
# STEP 5: DEFINE ML MODELS
# ==========================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ])
}

# ==========================================
# STEP 6: TRAIN AND EVALUATE MODELS
# ==========================================

results = []

print("\n========== MODEL TRAINING ==========")

for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test results
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluate Fail class (0)
    fail_precision = precision_score(
        y_test, y_pred,
        pos_label=0,
        zero_division=0
    )

    fail_recall = recall_score(
        y_test, y_pred,
        pos_label=0,
        zero_division=0
    )

    fail_f1 = f1_score(
        y_test, y_pred,
        pos_label=0,
        zero_division=0
    )

    # Evaluate Pass class (1)
    pass_recall = recall_score(
        y_test, y_pred,
        pos_label=1,
        zero_division=0
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Fail_Precision": fail_precision,
        "Fail_Recall": fail_recall,
        "Fail_F1_Score": fail_f1,
        "Pass_Recall": pass_recall
    })

    # Display classification report
    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    ))

    # Save trained model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        f"results/{filename}_confusion_matrix.png"
    )

    plt.close()

    print(f"{name} saved successfully.")

# ==========================================
# STEP 7: MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Fail_Recall",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

# ==========================================
# STEP 8: COMPLETION
# ==========================================

print("\n========== TRAINING COMPLETED ==========")
print("All four models trained successfully!")
print("Models saved inside models folder.")
print("Evaluation results saved inside results folder.")
