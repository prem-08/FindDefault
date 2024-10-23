# FindDefault: Credit Card Fraud Detection Project

## Overview

The FindDefault project is designed to predict fraudulent credit card transactions using machine learning models. The goal is to identify potentially fraudulent transactions in a highly imbalanced dataset. To achieve this, we apply multiple classification models: Logistic Regression, Random Forest, and XGBoost. We also handle data preprocessing, address class imbalance using SMOTE, perform feature engineering, and optimize models through hyperparameter tuning.

This project evaluates models using key performance metrics such as Accuracy, ROC AUC Score, Confusion Matrix, and F1 Score.

## Dataset

The dataset used in this project is the Credit Card Fraud Detection Dataset which contains anonymized transaction data, consisting of 31 columns:

* **Time**: Seconds elapsed between the transaction and the first transaction in the dataset.

* **V1 to V28**: Principal components obtained using PCA (details are not available).

* **Amount**: Transaction amount.

* **Class**: The target variable, where 1 indicates a fraudulent transaction and 0 indicates a legitimate one.

### Class Imbalance

The dataset contains a significant class imbalance, where only 0.17% of the transactions are fraudulent.

### Installation 

**Prerequisites:**

1. Python 3.x

2. Required Libraries:
   * numpy, pandas, matplotlib, seaborn, scikit - learn, xgboost, imbalanced - learn

## Project Steps

**1. Exploratory Data Analysis(EDA**

* Visualized the **Class Distribution** using countplot.

* Examined **Correlations** between features using a heatmap.

* Checked for missing values (none found).

**2. Data Preprocessing**

* **Feature Engineering:** Transformed the Time feature into the "Hour of the Day".

* **Scaling:** Standardized the Amount and Hour variables using StandardScaler.

**3. Handling Class Imbalance**

* Applied **SMOTE (Synthetic Minority Oversampling Technique)** to create synthetic samples for the minority class (fraudulent transactions) and balance the dataset.

**4. Model Training**

* Trained three models: **Logistic Regression, Random Forest,** and **XGBoost**.

* Split the data into 80% training and 20% testing sets.

**5. Model Evaluation**

* Evaluated models using metrics like **Accuracy, ROC AUC Score, F1 Score,** and **Confusion Matrix**.

**6. Hyperparameter Tuning**

* Performed Hyperparameter tuning for all models using **GridSearchCV** to find the optimal parameters and improve performance.

## Modeling and Evaluation

### Models Used:

**1.	Logistic Regression:** Provided a simple and interpretable baseline model.


**2.	Random Forest:** An ensemble model that improved classification performance.


**3.	XGBoost:** A powerful gradient-boosting model, which delivered the best performance.

### Evaluation Metrics:

* **Accuracy:** Measures the overall correctness of the model.

* **ROC AUC Score:** Reflects the model’s ability to distinguish between the positive and negative classes.

* ** F1 Score:** Balances precision and recall, focusing on the ability to correctly identify fraudulent transactions.

* **Confusion Matrix:** Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

## Results

### Initial Model Results:

* **Logistic Regression:** Performs decently but struggles with detecting complex patterns in fraud transactions.

* **Random Forest:** Shows significant improvement over Logistic Regression, capturing more fraud cases accurately.

* **XGBoost:** Delivers the highest performance in all metrics, making it the best model for this task.


### Hyperparameter Tuning Results:

* **XGBoost** remains the top-performing model after hyperparameter tuning, with its ability to fine-tune and generalize patterns in the data.

* **Random Forest** also sees improved performance with tuned parameters.

* **Logistic Regression** improves slightly but is outperformed by the other models.


## Conclusion

In this project, we explored multiple models for **credit card fraud detection** and concluded that:

* **XGBoost** was the most effective model in identifying fraudulent transactions, outperforming both Logistic Regression and Random Forest.

* Handling **class imbalance** with **SMOTE** and optimizing the models with **GridSearchCV** significantly enhanced model performance.

* Evaluation metrics such as **ROC AUC Score** and **F1 Score** proved to be more insightful than **Accuracy** in this highly imbalanced dataset.


This project demonstrates how advanced machine learning models and techniques can help identify fraudulent transactions and potentially save financial institutions from significant losses.

