# Predicting Movie Licensing Revenue Category

## Project Overview

This project implements an **end-to-end machine learning pipeline** to predict whether a movie
will fall into a **High** or **Low** revenue category for _Watch-It_, a media services company
whose revenue depends on advertising and brand integrations.

The solution is designed to support **data-driven content licensing decisions** by identifying
titles with higher expected revenue potential _before_ acquisition.

---

## Business Problem

Watch-It licenses movies to attract viewers and advertisers, but not all titles perform equally.
Poor licensing decisions can result in opportunity cost and reduced advertising revenue.

**Objective**  
Leverage historical movie metadata, audience engagement signals, ratings, and awards information
to predict the revenue category (`High` / `Low`) of new titles under consideration for licensing.

---

## Approach Summary

The solution follows a structured, industry-standard workflow:

1. **Data Integrity & Profiling**

   - Schema validation and sanity checks
   - Duplicate detection and missingness analysis

2. **Exploratory Data Analysis (EDA)**

   - Target distribution and class balance
   - Engagement, ratings, and awards relationships
   - Missingness patterns and feature skewness

3. **Data Cleaning & Standardisation**

   - Parsing numeric values stored as text
   - Normalizing categorical labels
   - Deterministic datetime parsing

4. **Feature Engineering**

   - Engagement quality metrics (rates and ratios)
   - Content lifecycle timing features
   - Log transformations for heavy-tailed variables

5. **Preprocessing Pipeline**

   - Median and mode imputation
   - Feature scaling
   - One-hot and multi-hot encoding for categorical data
   - Leakage-safe `scikit-learn` pipelines

6. **Model Selection**

   - 5-fold stratified cross-validation
   - Comparison of linear and ensemble models
   - Model selection based on F1-score

7. **Final Training & Prediction**
   - Best-performing model retrained on full training data
   - Predictions generated for the unseen test set

---

## Model Performance

- **Primary metric:** F1-score
- **Secondary metric:** Accuracy
- **Selected model:** Histogram Gradient Boosting

**Cross-validated performance:**

- F1-score ≈ **0.90**
- Accuracy ≈ **0.89**

These results indicate strong generalization performance and balanced classification across
revenue categories, making the model suitable for real-world decision support.

---

## Repository Contents

- `Predict_Movie_Licensing_Revenue.ipynb`  
  Narrative notebook containing EDA, feature engineering, modeling, and evaluation

- `src/`  
  Modular, production-style Python implementation of the full pipeline  
  (data loading, cleaning, feature engineering, preprocessing, modeling)

- `data/train.csv`  
  Training dataset

- `data/test.csv`  
  Unlabeled test dataset

- `submissions.csv`  
  Final prediction output (High / Low revenue category)

- `requirements.txt`  
  Python dependencies required to reproduce the environment

---

## How to Run

### Option 1: End-to-End Pipeline (Recommended)

```bash
python -m src.main --data_dir data --out_dir artifacts
```

**Outputs:**

- artifacts/submissions.csv
- EDA artifacts (plots and summary tables)

---

**Key Takeaways**

- Audience engagement intensity and quality are strong predictors of revenue performance
- Missing engagement data can be informative and should not be discarded blindly
- Multi-value categorical features (genres, countries, languages) require careful encoding
- Leakage-safe pipelines are critical for reliable model evaluation

---

**Author**

Olalemi John

Data Scientist
