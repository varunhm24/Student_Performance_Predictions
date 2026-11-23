
import joblib
import os

# Create models directory if needed
os.makedirs("models", exist_ok=True)

# Load the trained Logistic Regression model
model = joblib.load(
    "models/logistic_regression.pkl"
)

# Save it as the final project model
joblib.dump(
    model,
    "models/student_performance_model.pkl"
)

print("Final model saved successfully!")
print("File: models/student_performance_model.pkl")
