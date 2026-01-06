# 📊 Diabetes Hospital Readmission Prediction

## Overview
Hospital readmissions are a major challenge in modern healthcare, contributing to increased costs, resource strain, and poorer patient outcomes. This project applies data science and machine learning techniques to **predict 30-day hospital readmission** for diabetic patients using a real-world clinical dataset.

The goal is to demonstrate an end-to-end healthcare analytics workflow: data preparation, exploration, modelling, and evaluation, grounded in academic best practice.

---

## Dataset
- **Source:** UCI Machine Learning Repository – *Diabetes 130-US hospitals dataset*
- **Records:** ~100,000 patient encounters  
- **Features:** Demographics, diagnoses, medications, procedures, length of stay  
- **Target variable:** Readmission status (`<30`, `>30`, `No`)

This dataset is intentionally **messy and realistic**, reflecting real clinical data quality challenges such as missing values, encoded diagnoses, and class imbalance.

---

## Problem Formulation
This project frames readmission prediction as a **supervised classification problem**, focusing on identifying patients at risk of being readmitted within 30 days of discharge.

Early identification enables:
- targeted post-discharge interventions  
- reduced avoidable readmissions  
- improved healthcare efficiency  

---

## Methodology
The project follows a structured data science workflow inspired by **CRISP-DM**, including:

1. Data understanding & quality assessment  
2. Data cleaning and preprocessing  
3. Exploratory data analysis (EDA)  
4. Feature selection & encoding  
5. Predictive modelling  
6. Model evaluation and comparison  

All modelling decisions are justified with reference to healthcare analytics literature.

---

## Models Used
- Logistic Regression  
- Decision Tree  
- Random Forest  

Models are compared using appropriate evaluation metrics, with attention paid to **interpretability**, which is critical in healthcare contexts.

---

## Tools & Technologies
- **Python**
- pandas, NumPy  
- scikit-learn  
- matplotlib / seaborn  

All tools were selected for their reliability, transparency, and widespread adoption in healthcare data science.

---

## Key Findings
- Readmission is influenced by factors such as prior hospital utilisation, length of stay, and diagnosis complexity.  
- Ensemble models outperform simpler classifiers but introduce interpretability trade-offs.  
- Data quality and feature engineering significantly impact model performance.  

---

## Ethical Considerations
- Patient data is anonymised and sourced from an open dataset.  
- Predictions are intended for **decision support**, not clinical replacement.  
- Bias, fairness, and model interpretability are explicitly considered.  

---
## Author
**David Arubuike**  
