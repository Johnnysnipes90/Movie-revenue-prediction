# Predicting Movie Licensing Revenue Category

## Project Overview

This project builds an **end-to-end machine learning pipeline** to predict whether a movie
will fall into a **High** or **Low** revenue category for Watch-It, a media services company
whose revenue depends on advertising and brand integrations.

The goal is to support **data-driven licensing decisions** by identifying titles
with higher expected revenue potential before acquisition.

---

## Business Problem

Watch-It licenses movies to attract viewers and advertisers.
Not all titles perform equally, and licensing decisions involve financial risk.

**Objective:**  
Use historical movie metadata, engagement signals, ratings, and awards data to predict
the revenue category (`High` / `Low`) of new titles.

---

## Approach Summary

The solution follows an end-to-end, industry-standard workflow:

1. **Data Integrity & Profiling**
   - Schema validation, duplicate checks, missingness analysis
2. **Exploratory Data Analysis (EDA)**
   - Target balance
   - Engagement and ratings impact
   - Missingness patterns and skewness analysis
3. **Data Cleaning & Standardisation**
   - Parsing numeric values stored as text
   - Normalizing categorical labels
   - Robust datetime handling
4. **Feature Engineering**
   - Engagement quality metrics (rates, ratios)
   - Lifecycle timing features
   - Log-transformed heavy-tailed variables
5. **Preprocessing Pipeline**
   - Imputation, scaling, one-hot and multi-hot encoding
   - Leakage-safe sklearn pipelines
6. **Model Selection**
   - 5-fold stratified cross-validation
   - Comparison of linear and ensemble models
7. **Final Training & Prediction**
   - Best-performing model retrained on full data
   - Predictions generated for the unseen test set

---

## Model Performance

- **Evaluation metric:** F1-score (primary), Accuracy (secondary)
- **Best model:** Histogram Gradient Boosting
- **Cross-validated performance:**
  - F1-score ≈ **0.90**
  - Accuracy ≈ **0.89**

This performance demonstrates strong generalization and balanced classification
across revenue categories, making the model suitable for decision support.

---

## Repository Contents

- `Predict_Movie_Licensing_Revenue.ipynb`  
  End-to-end analysis, EDA, feature engineering, modeling, and evaluation
- `train.csv`  
  Training dataset
- `test.csv`  
  Test dataset (unlabeled)
- `submissions.csv`  
  Final prediction output
- `requirements.txt`  
  Python dependencies to reproduce the environment

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Predict_Movie_Licensing_Revenue.ipynb
```

The notebook is designed to run end-to-end without modification.

---

**Key Takeaways**

- Engagement intensity and quality are strong predictors of revenue performance
- Missing engagement data can be informative and should not be discarded blindly
- Multi-value categorical features (genres, countries, languages) require careful encoding
- Leakage-safe pipelines are critical for reliable model evaluation

---

**Author**

Olalemi John

Data Scientist
