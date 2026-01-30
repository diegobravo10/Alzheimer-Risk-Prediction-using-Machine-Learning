
# 🧠 Alzheimer Risk Prediction using Machine Learning

An intelligent predictive system that estimates Alzheimer’s disease risk using clinical, demographic, lifestyle, and cognitive assessment data.  
The project uses **XGBoost** for modeling and **FastAPI** for real-time prediction through a web interface.

> ⚠️ This system provides risk estimation — it is not a medical diagnosis tool.

---

## Features

- ✅ Machine Learning model based on **XGBoost**
- ✅ Clinical + lifestyle + cognitive variables
- ✅ Feature engineering pipeline
- ✅ MLflow model tracking (optional)
- ✅ FastAPI REST service
- ✅ Interactive web form frontend
- ✅ Real-time risk prediction
- ✅ Ready for retraining and model versioning

---

## 🧠 Model

The prediction model was trained using:

- Demographic data
- Medical history
- Lifestyle factors
- Clinical measurements
- Cognitive and functional assessments
- Symptom indicators

Algorithm used:

```

XGBoost Classifier

```

Derived features include:

- cognitive decline score
- vascular risk score
- symptom count
- lifestyle score
- age interactions
- clinical ratios

---

##  Project Structure

```

fastapi-alzheimer/
│
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── notebooks/
│   └── alzheimer-disease-prediction-exploratory-analysis.ipynb
│   └── transformation-and-processing-of-variables.ipynb
│   └── training-with-xgboost.ipynb
│   └── prediction-new-patients
│   └── ....
└── README.md

````

---

## ▶️ Run the API

```bash
uvicorn app:app --reload
```

Open browser:

```
http://127.0.0.1:8000
```

---

## 📊 Input Variables

The model uses:

* Age
* Gender
* Ethnicity
* Education
* BMI
* Physical activity
* Diet quality
* Sleep quality
* Blood pressure
* Cholesterol measures
* MMSE
* Functional assessment
* ADL
* Medical history flags
* Cognitive symptoms

The web interface includes guided ranges for each field.

---

## 🔬 Methodology

1. Data exploration
2. Cleaning and preprocessing
3. Feature engineering
4. Model training (XGBoost)
5. Evaluation
6. Model versioning
7. API deployment with FastAPI
8. Web interface integration

---

## Dependencies

Main libraries:

* fastapi
* uvicorn
* pandas
* scikit-learn
* xgboost
* mlflow

---

## Disclaimer

This project is for educational and research purposes only.
It does **not** replace medical evaluation or diagnosis.

---

## Authors

Diego Bravo & Ariel Paltan — Computer Science Students

