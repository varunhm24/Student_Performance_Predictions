# Student Performance Prediction System

**live demo**:https://student-performance-predictions-2.onrender.com/

## Project Overview

The **Student Performance Prediction System** is a Machine Learning-based web application designed to predict students' academic performance based on various academic and personal factors.

The system analyzes student-related information and uses a trained Machine Learning model to predict whether a student is likely to **Pass or Fail**. It helps identify students who may need additional academic support and enables data-driven educational decision-making.

## Objectives

- Predict student academic performance using Machine Learning.
- Analyze important factors influencing student results.
- Identify students who may be at academic risk.
- Provide quick and accurate predictions through a web interface.
- Support better academic monitoring and decision-making.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning model development |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Flask | Web application development |
| HTML | Web page structure |
| CSS | Web page styling |
| Joblib | Saving and loading trained models |

## Machine Learning Workflow

1. Dataset Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Selection
5. Model Training
6. Model Evaluation
7. Best Model Selection
8. Model Saving
9. Flask Web Application Development
10. Student Performance Prediction

## Features

- Student academic performance prediction.
- Pass/Fail classification.
- User-friendly web interface.
- Data preprocessing and feature handling.
- Machine Learning model integration.
- Fast prediction results.
- Data-driven academic analysis.

## Project Structure

```text
Student_Performance_Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── dataset/
│   └── student_data.csv
│
├── models/
│   └── best_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── charts/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── other_visualizations.png
│
└── README.md
```

*Note: The folder structure above is a reference. Adjust it according to the actual files in your repository.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Student_Performance_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Train the Machine Learning Model

```bash
python train_model.py
```

### 7. Run the Flask Application

```bash
python app.py
```

### 8. Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## Prediction Output

The application accepts student-related input data and generates a predicted academic result.

- **Pass:** The model predicts that the student is likely to pass.
- **Fail:** The model predicts that the student may be at risk of failing.

## Applications

- Educational institutions.
- Academic performance monitoring.
- Student progress analysis.
- Early identification of academically at-risk students.
- Academic decision support systems.

## Future Enhancements

- Integration with a larger student dataset.
- Addition of more Machine Learning algorithms.
- Student performance visualization dashboard.
- Student-wise performance history.
- Deployment on cloud platforms.
- Integration of student improvement recommendations.

## Author

**Varun H M**

Engineering Student  
Channabasaveshwara Institute of Technology

## License

This project is developed for educational and academic purposes. You are welcome to explore and learn from the implementation.
