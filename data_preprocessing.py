import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency

class SmokingDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_final = None

    def load_data(self):
        """Loads and selects key columns."""
        print("📂 Loading Dataset...")
        self.df = pd.read_csv(self.file_path)
        selected_cols = ['age', 'weight(kg)', 'Gtp', 'HDL', 'gender', 'smoking']
        self.df = self.df[selected_cols].copy()
        print(f"✅ Data Loaded. Shape: {self.df.shape}")

    def perform_eda_and_stats(self):
        """Performs statistical tests (Mann-Whitney & Chi-Square)."""
        print("\n📊 Performing Statistical Hypothesis Testing...")
        
        # Numerical Tests
        numerical_features = ['age', 'weight(kg)', 'Gtp', 'HDL']
        smokers = self.df[self.df['smoking'] == 1]
        non_smokers = self.df[self.df['smoking'] == 0]
        
        for col in numerical_features:
            stat, p_val = mannwhitneyu(smokers[col], non_smokers[col])
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            print(f"   - {col}: p-value={p_val:.4e} ({sig})")

        # Categorical Test
        contingency = pd.crosstab(self.df['gender'], self.df['smoking'])
        chi2, p_val, _, _ = chi2_contingency(contingency)
        print(f"   - Gender: p-value={p_val:.4e} ({'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'})")

    def transform_data(self):
        """Applies Log and Sqrt transformations."""
        print("\n🔄 Applying Log and Sqrt Transformations...")
        self.df['Gtp_log'] = np.log1p(self.df['Gtp'])
        self.df['HDL_log'] = np.log1p(self.df['HDL'])
        self.df['weight_sqrt'] = np.sqrt(self.df['weight(kg)'])
        
        final_cols = ['age', 'gender', 'weight_sqrt', 'Gtp_log', 'HDL_log', 'smoking']
        self.df_final = self.df[final_cols].copy()
        
        # One-Hot Encoding
        self.df_final = pd.get_dummies(self.df_final, columns=['gender'], drop_first=True)
        print("✅ Data Transformation Complete.")
        return self.df_final

if __name__ == "__main__":
    # Test run
    processor = SmokingDataProcessor("smoking.csv")
    processor.load_data()
    processor.perform_eda_and_stats()
    df_clean = processor.transform_data()
    print(df_clean.head())