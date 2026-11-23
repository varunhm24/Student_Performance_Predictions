
import pandas as pd
import numpy as np

# Reproducible simulated dataset
np.random.seed(42)

# Number of student records
number_of_students = 500

# Generate student academic information
student_id = [
    f"STU{i:03d}"
    for i in range(1, number_of_students + 1)
]

study_hours = np.round(
    np.random.uniform(0, 10, number_of_students), 1
)

attendance = np.random.randint(
    30, 101, number_of_students
)

previous_marks = np.random.randint(
    0, 101, number_of_students
)

assignments = np.random.randint(
    0, 101, number_of_students
)

internal_marks = np.random.randint(
    0, 31, number_of_students
)

# Calculate an illustrative academic score
academic_score = (
    previous_marks * 0.25
    + assignments * 0.20
    + (internal_marks / 30 * 100) * 0.30
    + attendance * 0.15
    + study_hours * 10 * 0.10
)

# Add variation to simulated outcomes
noise = np.random.normal(
    0, 8, number_of_students
)

final_score = academic_score + noise

# Assign Pass or Fail
final_result = np.where(
    final_score >= 40,
    "Pass",
    "Fail"
)

# Create DataFrame
df = pd.DataFrame({
    "Student_ID": student_id,
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Assignments": assignments,
    "Internal_Marks": internal_marks,
    "Final_Result": final_result
})

# Save dataset
df.to_csv(
    "dataset/student_performance.csv",
    index=False
)

# Display dataset information
print("Student dataset created successfully!")

print("\nFirst 5 student records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nPass/Fail distribution:")
print(df["Final_Result"].value_counts())

print("\nDataset saved in dataset/student_performance.csv")

