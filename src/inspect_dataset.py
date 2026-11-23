
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/student_performance.csv")

print("\n========== DATASET INSPECTION ==========")

# First five records
print("\n1. First five records:")
print(df.head())

# Dataset size
print("\n2. Dataset shape:")
print(df.shape)

# Column names
print("\n3. Column names:")
print(df.columns.tolist())

# Dataset information
print("\n4. Dataset information:")
df.info()

# Statistical summary
print("\n5. Statistical summary:")
print(df.describe())

# Missing values
print("\n6. Missing values:")
print(df.isnull().sum())

# Duplicate records
print("\n7. Duplicate records:")
print(df.duplicated().sum())

# Unique student IDs
print("\n8. Unique student IDs:")
print(df["Student_ID"].nunique())

# Result distribution
print("\n9. Final result distribution:")
print(df["Final_Result"].value_counts())

print("\n========== INSPECTION COMPLETED ==========")
