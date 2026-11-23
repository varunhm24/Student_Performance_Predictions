
import pandas as pd

# Load cleaned dataset
df = pd.read_csv(
    "dataset/student_performance_cleaned.csv"
)

# Select input features
X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignments",
        "Internal_Marks"
    ]
]

# Select target variable
y = df["Final_Result"].map({
    "Fail": 0,
    "Pass": 1
})

# Display feature information
print("========== FEATURE SELECTION ==========")

print("\nInput features:")
print(X.head())

print("\nTarget variable:")
print(y.head())

print("\nInput feature shape:", X.shape)
print("Target variable shape:", y.shape)

print("\nFeature names:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeature selection completed successfully!")
