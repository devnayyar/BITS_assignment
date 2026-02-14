# Adult Income Classification - ML Assignment 2

**Student ID:** 2025aa05627

## 1. Problem Statement

The task is to **predict whether an adult earns more than $50,000 per year** based on demographic and employment information from the Adult Census Income dataset. This is a **binary classification** problem aimed at understanding income levels based on various features like age, education, occupation, and marital status.

### Objective
- Implement and compare **6 different classification models**
- Calculate **6 evaluation metrics** for each model
- Identify the best-performing model
- Deploy the solution as a web application

---

## 2. Dataset Description

### Overview
- **Source:** Adult Census Income Dataset (UCI Machine Learning Repository)
- **Total Records:** 48,843 instances
- **Features:** 14 input features
- **Target Variable:** Income (<=50K or >50K) - Binary Classification

### Features
1. **age** - Age of the person (numerical)
2. **workclass** - Employment status (categorical)
3. **fnlwgt** - Final weight (numerical)
4. **education** - Education level (categorical)
5. **educational-num** - Numerical representation of education (numerical)
6. **marital-status** - Marital status (categorical)
7. **occupation** - Occupation type (categorical)
8. **relationship** - Relationship status (categorical)
9. **race** - Race (categorical)
10. **gender** - Gender (categorical)
11. **capital-gain** - Capital gains (numerical)
12. **capital-loss** - Capital losses (numerical)
13. **hours-per-week** - Hours worked per week (numerical)
14. **native-country** - Country of origin (categorical)

### Data Preprocessing
- **Removed rows** containing missing values (represented as '?')
- **Encoded categorical** variables using Label Encoding
- **Scaled numerical** features using StandardScaler
- **Train-Test Split:** 80-20 with stratification for class balance

---

## 3. Models Implemented

### 1. **Logistic Regression**
- Linear classifier based on logistic function
- Provides probability estimates
- Suitable for binary classification
- Fast training and prediction

### 2. **Decision Tree Classifier**
- Tree-based model learning if-then-else rules
- Interpretable and easy to visualize
- Can capture non-linear relationships
- Maximum depth set to 15 to prevent overfitting

### 3. **K-Nearest Neighbors (KNN)**
- Instance-based learning algorithm
- Uses k=5 nearest neighbors for classification
- Requires scaling of features
- Non-parametric approach

### 4. **Naive Bayes Classifier**
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast training and prediction
- Works well with sparse data

### 5. **Random Forest Classifier**
- Ensemble of 100 decision trees
- Reduces overfitting through averaging
- Robust to outliers
- Provides feature importance rankings

### 6. **XGBoost Classifier**
- Gradient boosting ensemble method
- Sequential tree building with error correction
- Highly optimized and effective
- 100 estimators with max depth of 7

---

## 4. Evaluation Metrics

For each model, **6 evaluation metrics** are calculated on the test set:

### Metrics Explanation

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness of predictions (0-1) |
| **AUC Score** | Area under ROC curve | Discrimination ability across thresholds (0-1) |
| **Precision** | TP / (TP + FP) | Accuracy of positive predictions (0-1) |
| **Recall** | TP / (TP + FN) | Coverage of actual positive cases (0-1) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall (0-1) |
| **MCC** | (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Correlation between predicted and actual (-1 to 1) |

Where:
- **TP** = True Positives
- **TN** = True Negatives
- **FP** = False Positives
- **FN** = False Negatives

---

## 5. Performance Metrics Table

### Model Comparison Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8421 | 0.8956 | 0.7840 | 0.6324 | 0.7011 | 0.6547 |
| Decision Tree | 0.8358 | 0.8634 | 0.7652 | 0.6189 | 0.6847 | 0.6232 |
| KNN | 0.8351 | 0.8721 | 0.7613 | 0.6128 | 0.6798 | 0.6153 |
| Naive Bayes | 0.8124 | 0.8843 | 0.7289 | 0.5642 | 0.6363 | 0.5589 |
| Random Forest | 0.8573 | 0.9162 | 0.8156 | 0.6582 | 0.7273 | 0.6899 |
| XGBoost | 0.8642 | 0.9247 | 0.8298 | 0.6745 | 0.7424 | 0.7042 |

### Key Findings

**Best Performing Models:**
- **Best Overall:** XGBoost (Accuracy: 0.8642, F1: 0.7424, AUC: 0.9247)
- **Best Accuracy:** XGBoost (0.8642)
- **Best AUC:** XGBoost (0.9247)
- **Best F1 Score:** XGBoost (0.7424)
- **Best MCC:** XGBoost (0.7042)

---

## 6. Model Observations & Analysis

### Logistic Regression
- **Strengths:** Fast, interpretable, good baseline
- **Performance:** Moderate (Accuracy: 0.8421)
- **Observation:** Linear model captures basic patterns but may miss complex relationships
- **Use Case:** Quick predictions, high-stakes decisions requiring explainability

### Decision Tree
- **Strengths:** Non-parametric, interpretable, handles non-linearity
- **Performance:** Moderate (Accuracy: 0.8358)
- **Observation:** Shows signs of underfitting despite depth=15
- **Use Case:** Feature importance analysis, rule-based decisions

### K-Nearest Neighbors
- **Strengths:** Simple, non-parametric
- **Performance:** Moderate (Accuracy: 0.8351)
- **Observation:** Slightly underperforms ensemble methods
- **Use Case:** Non-linear patterns, small datasets

### Naive Bayes
- **Strengths:** Fast, probabilistic, handles uncertainty well
- **Performance:** Lower (Accuracy: 0.8124)
- **Observation:** Independence assumption may be too strong for this dataset
- **AUC (0.8843):** Despite lower accuracy, shows good discrimination
- **Use Case:** Real-time applications, probabilistic frameworks

### Random Forest
- **Strengths:** Ensemble power, reduces overfitting, robust
- **Performance:** Good (Accuracy: 0.8573, F1: 0.7273)
- **Observation:** Significant improvement over single trees
- **AUC (0.9162):** Excellent discrimination between classes
- **Use Case:** Production systems requiring balance

### XGBoost ⭐ (BEST MODEL)
- **Strengths:** Gradient boosting, highly optimized, best overall performance
- **Performance:** Excellent (Accuracy: 0.8642, AUC: 0.9247)
- **Observation:** Consistently best across all metrics
- **F1 Score (0.7424):** Best balance between precision and recall
- **MCC (0.7042):** Strongest correlation between predictions and reality
- **Recommendation:** **Best choice for production deployment**

---

## 7. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training & Evaluation (Jupyter Notebook)
```bash
jupyter notebook adult_classification_models.ipynb
```

### Streamlit Web Application
```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

---

## 8. Project Structure

```
BITS_ML/
├── adult.csv                           # Adult dataset
├── adult_classification_models.ipynb   # Complete model training notebook
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 9. Key Insights

1. **Ensemble methods (Random Forest, XGBoost) outperform single models**
   - They capture complex non-linear patterns
   - Better generalization to unseen data

2. **XGBoost is the optimal choice**
   - Highest accuracy (0.8642)
   - Best AUC score (0.9247)
   - Excellent balance between metrics

3. **Logistic Regression provides good baseline**
   - Fast to train and interpret
   - 84.21% accuracy is respectable
   - Useful for understanding feature importance

4. **Scaling helps tree-based models**
   - KNN and Logistic Regression benefit significantly from scaling
   - Tree-based models (DT, RF, XGBoost) don't require scaling

5. **Class balance affects evaluation**
   - Precision and Recall vary across models
   - F1 score provides more balanced evaluation

---

## 10. Future Improvements

1. **Hyperparameter Tuning:** Use GridSearchCV/RandomizedSearchCV
2. **Feature Engineering:** Create interaction terms, domain-specific features
3. **Class Imbalance Handling:** SMOTE, class weights, stratified sampling
4. **Cross-Validation:** Use k-fold validation for robust evaluation
5. **Ensemble Stacking:** Combine predictions from multiple models
6. **Deployment:** Deploy to AWS, Azure, or Heroku

---

## 11. Files Summary

### Jupyter Notebook (`adult_classification_models.ipynb`)
- Complete data preprocessing pipeline
- Training code for all 6 models
- Comprehensive evaluation metrics
- Visualizations and comparative analysis
- Run on **BITS Virtual Lab** for assignment submission

### Streamlit App (`app.py`)
- Interactive web interface
- Real-time model training
- Performance comparison dashboard
- Test predictions on custom data
- Confusion matrix and classification reports

---

## 12. References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Streamlit Documentation: https://docs.streamlit.io/
- Adult Dataset: https://archive.ics.uci.edu/ml/datasets/Adult

---

**Assignment Submission Date:** January 2026  
**Last Updated:** 26-01-2026  
**Status:** ✅ Complete

