# Developer Salary Prediction 

<!-- Repository Stats -->
![GitHub repo size](https://img.shields.io/github/repo-size/nyambura-tech/Dev-salary-prediction)
![GitHub last commit](https://img.shields.io/github/last-commit/nyambura-tech/Dev-salary-prediction)
![GitHub stars](https://img.shields.io/github/stars/nyambura-tech/Dev-salary-prediction)
![GitHub forks](https://img.shields.io/github/forks/nyambura-tech/Dev-salary-prediction)
![License](https://img.shields.io/github/license/nyambura-tech/Dev-salary-prediction)

<!-- Tech Stack -->
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-blueviolet)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plot-orange)

<!-- Project Status -->
![Status](https://img.shields.io/badge/Status-Active-success)
![ML Models](https://img.shields.io/badge/Models-RandomForest%2C%20GradientBoosting%2C%20LinearRegression-blue)

# Developer Salary Prediction

The Project was focused on predicting the salary of Developer with reference to various factors such Languages Worked with. and Geographic location and later more features were added. The model is trained on developer survey data to understand how different factors influence salary levels.

---

## Project Overview

This project uses data from the **Stack Overflow Developer Survey** to build a machine learning model capable of predicting a developer's annual salary based on demographic, educational, employment, and technical experience information.

**Dataset:** Stack Overflow Developer Survey 2025

---

## Project Workflow

The project follows a complete machine learning workflow:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Model Serialization for Deployment

**Final Model:** XGBoost Regressor integrated into a reusable Scikit-Learn pipeline.

---

## Dataset

### Source
Stack Overflow Developer Survey Dataset

### Target Variable
**`ConvertedCompYearly`** - Annual compensation converted to USD

### Salary Filtering
To improve model quality, salaries are filtered to:
- Minimum: `>= 10,000`
- Maximum: `<= 500,000`

---

## Exploratory Data Analysis (`01_eda.ipynb`)

### Purpose
Understand the dataset before modeling.

### Tasks Performed
- Loading raw survey data
- Exploring dataset dimensions
- Inspecting missing values
- Analyzing salary distribution
- Examining developer demographics
- Exploring education levels
- Investigating employment status
- Analyzing programming language usage
- Identifying outliers
- Salary trends by experience

---

## Data Preprocessing (`preprocessing.ipynb`)

### Purpose
Prepare a cleaned dataset for machine learning.

### Feature Selection
Selected features:
- `Country`
- `YearsCode`
- `EdLevel`
- `Employment`
- `LanguageHaveWorkedWith`

---

## Source Code Structure

### `preprocess.py`
Reusable data cleaning functions

- Removing missing salaries & unrealistic salary values
- Education standardization (e.g., "Bachelor's degree" → "Bachelor's")
- Employment cleaning (Full-time, Freelance/Self-employed, Student, Other)
- Language feature engineering (converts language lists to counts)
  - Example: `Python;SQL;JavaScript;HTML/CSS` → `4`
- Country grouping (keeps top 15 countries, others → "Other")
- Automatic feature detection (returns categorical & numeric columns)

---

### `evaluation.py`
Model performance utilities

#### Metrics Computed
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

#### Visualization Functions
- Actual vs Predicted Plot
- Actual Salary vs Predicted Salary

---

### `train.py`
End-to-end training script

#### Main Function
1. Load and clean data
2. Save processed dataset
3. Split data (train/test)
4. Build preprocessing pipeline
5. Train XGBoost model
6. Evaluate performance
7. Save trained pipeline

---



