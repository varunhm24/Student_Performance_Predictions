
import pandas as pd

# Load original dataset
df = pd.read_csv("dataset/student_performance.csv")

print("========== DATA PREPROCESSING ==========")

# Step 1: Original dataset information
print("\nOriginal dataset shape:", df.shape)

# Step 2: Remove duplicate records
duplicate_count = df.duplicated().sum()
print("\nDuplicate records found:", duplicate_count)

df = df.drop_duplicates()

# Step 3: Handle missing Student_ID and Final_Result
df = df.dropna(subset=["Student_ID", "Final_Result"])

# Step 4: Handle missing numerical values
numeric_columns = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignments",
    "Internal_Marks"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Step 5: Validate target labels
valid_results = ["Pass", "Fail"]

invalid_results = ~df["Final_Result"].isin(valid_results)

if invalid_results.any():
    raise ValueError("Invalid values found in Final_Result!")

# Step 6: Validate numerical ranges
valid_ranges = {
    "Study_Hours": (0, 10),
    "Attendance": (30, 100),
    "Previous_Marks": (0, 100),
    "Assignments": (0, 100),
    "Internal_Marks": (0, 30)
}

for column, (minimum, maximum) in valid_ranges.items():
    invalid = ~df[column].between(minimum, maximum)

    if invalid.any():
        raise ValueError(
            f"Invalid values found in {column}"
        )

# Step 7: Prepare input features and target
X = df.drop(columns=["Student_ID", "Final_Result"])
y = df["Final_Result"].map({
    "Fail": 0,
    "Pass": 1
})

# Step 8: Save cleaned dataset
df.to_csv(
    "dataset/student_performance_cleaned.csv",
    index=False
)

print("\nCleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nInput feature columns:")
print(X.columns.tolist())

print("\nTarget variable:")
print(y.name)

print("\nTarget encoding:")
print(y.value_counts().to_dict())

print("\nCleaned dataset saved successfully!")
print("File: dataset/student_performance_cleaned.csv")
