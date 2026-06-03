# Heart Disease Prediction System

A machine learning web application that predicts the likelihood of heart disease based on patient medical parameters. Built using **K-Nearest Neighbors (KNN)** and deployed using **Flask + Render**.

---

## Live Demo

👉 https://heart-disease-prediction-using-knn.onrender.com

---

##  Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection can help save lives. This project uses machine learning to predict whether a patient is at risk based on medical inputs.

---

##  Tech Stack

- Python 
- Flask (Backend API)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- HTML, CSS, JavaScript (Frontend)
- Render (Deployment)

---

## Model Details

- Algorithm: K-Nearest Neighbors (KNN)
- Data Preprocessing: Scaling + Encoding
- Output: Binary Classification (0 = No Disease, 1 = Risk)

### Features Used:
- Age  
- Resting Blood Pressure  
- Cholesterol  
- Max Heart Rate  
- Oldpeak  
- Sex  
- Chest Pain Type  
- ST Slope  
- Exercise Angina  

---

##  Workflow

1. User enters input in web form  
2. Frontend sends JSON request to Flask API  
3. Data is preprocessed (scaling + encoding)  
4. ML model predicts output  
5. Result is returned and displayed on UI  

---

## 📡 API Endpoint

### POST `/predict`

#### Request Body:
```json
{
  "Age": 55,
  "RestingBP": 140,
  "Cholesterol": 250,
  "MaxHR": 120,
  "Oldpeak": 2.0,
  "Sex_M": 1,
  "ChestPainType_ATA": 0,
  "ST_Slope_Up": 1,
  "ExerciseAngina_Y": 0
}
