# EV Battery Remaining Useful Life (RUL) Prediction

## Overview

The growing adoption of Electric Vehicles (EVs) has increased the importance of battery health monitoring and predictive maintenance. This project focuses on predicting the Remaining Useful Life (RUL) of EV batteries using Machine Learning techniques.

The objective is to estimate how much useful life remains in a battery based on historical operational parameters, enabling better maintenance planning, improved reliability, and reduced operational costs.

---

## Problem Statement

Battery degradation directly affects the performance, range, and safety of electric vehicles. Accurately predicting the Remaining Useful Life (RUL) of batteries can help manufacturers and users make informed maintenance and replacement decisions.

This project develops and compares multiple Machine Learning regression models to predict battery RUL from battery performance data.

---

## Dataset

The dataset contains battery operational parameters and corresponding Remaining Useful Life (RUL) values.

### Features Used

- Discharge Time (s)
- Decrement 3.6–3.4V (s)
- Max Voltage Discharge (V)
- Min Voltage Charge (V)
- Time at 4.15V (s)
- Time Constant Current (s)
- Charging Time (s)
- Other battery health indicators

### Target Variable

- RUL (Remaining Useful Life)

---

## Project Workflow

### 1. Data Preprocessing

- Dataset inspection and validation
- Missing value analysis
- Data cleaning
- Removal of invalid summary row

### 2. Exploratory Data Analysis (EDA)

- Statistical analysis
- Correlation heatmap
- Outlier detection using boxplots

### 3. Outlier Treatment

Outliers were handled using the Interquartile Range (IQR) Capping technique to improve model stability and performance.

### 4. Feature Scaling

- StandardScaler was applied to normalize feature values.
- Data was split into training and testing sets using an 80:20 ratio.

---

## Machine Learning Models Implemented

The following regression algorithms were trained and evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Support Vector Regression (SVR)
7. K-Nearest Neighbors (KNN) Regressor
8. XGBoost Regressor

---

## Hyperparameter Tuning

GridSearchCV with 5-fold Cross Validation was used for:

- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

This helped identify the optimal model configurations and improve prediction accuracy.

---

## Evaluation Metrics

Model performance was evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Visualization

The project includes:

- Correlation Heatmap
- Outlier Detection Boxplots
- Feature Importance Plot (XGBoost)
- Actual vs Predicted RUL Plot
- Model Performance Comparison

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost

---

## Key Learnings

Through this project, I gained practical experience in:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Outlier Handling
- Regression Modeling
- Hyperparameter Optimization
- Model Evaluation
- Predictive Maintenance Applications

---
