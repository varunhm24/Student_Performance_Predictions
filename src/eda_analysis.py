
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv(
    "dataset/student_performance_cleaned.csv"
)

# Create results folder if missing
os.makedirs("results", exist_ok=True)

# Convert Pass/Fail into numeric values for analysis
df["Result_Encoded"] = df["Final_Result"].map({
    "Fail": 0,
    "Pass": 1
})

sns.set_theme(style="whitegrid")

# 1. Pass and Fail distribution
plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="Final_Result"
)
plt.title("Student Pass and Fail Distribution")
plt.xlabel("Final Result")
plt.ylabel("Number of Students")
plt.savefig("results/result_distribution.png")
plt.show()

# 2. Study hours vs final result
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Final_Result",
    y="Study_Hours"
)
plt.title("Study Hours vs Student Performance")
plt.savefig("results/study_hours_analysis.png")
plt.show()

# 3. Attendance vs final result
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Final_Result",
    y="Attendance"
)
plt.title("Attendance vs Student Performance")
plt.savefig("results/attendance_analysis.png")
plt.show()

# 4. Previous marks vs final result
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Final_Result",
    y="Previous_Marks"
)
plt.title("Previous Marks vs Student Performance")
plt.savefig("results/previous_marks_analysis.png")
plt.show()

# 5. Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(9, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Student Performance Correlation Heatmap")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.show()

print("EDA analysis completed successfully!")
print("Graphs saved inside the results folder.")