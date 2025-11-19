# 🏥 Rigorous Comparative Analysis of ML Models for Smoking Prediction

### 🧪 Project Overview
This study evaluates the efficacy of three distinct machine learning algorithms (**KNN, Logistic Regression, Random Forest**) in classifying smoking status using physiological health markers.
*   **Dataset:** 55,692 records (Kaggle Health Screening Data).
*   **Goal:** Predict binary smoking status (0/1) based on Age, Weight, Gender, GTP, and HDL levels.

### 📊 Key Findings
*   **Champion Model:** Random Forest (n=400) achieved the highest performance with an **AUC of 0.85** and **Macro F1-Score of 0.75**.
*   **Statistical Significance:** Mann-Whitney U and Chi-Square tests confirmed that all 5 selected predictors were statistically significant (p < 0.05).
*   **Feature Importance:** `Gtp_log` and `gender` were identified as the strongest predictors of smoking habits.

### ⚙️ Methodology
1.  **Exploratory Data Analysis (EDA):** Q-Q plots revealed skewness in GTP and HDL, which were corrected using **Log1p transformations**.
2.  **Preprocessing:** applied Square-Root transform to Weight and One-Hot Encoding to Gender.
3.  **Model Selection:**
    *   **KNN:** Optimized via Elbow Method (Best k=15).
    *   **Random Forest:** Optimized via Grid Search (Best n_estimators=400).
4.  **Evaluation:** Stratified 10-Fold Cross-Validation.

### 🛠️ Tech Stack
*   **Python** (Pandas, NumPy, Scikit-Learn)
*   **Stats:** SciPy (Hypothesis Testing)
*   **Viz:** Matplotlib, Seaborn

### 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python main.py`
