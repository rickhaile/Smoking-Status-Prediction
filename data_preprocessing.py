import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


class SmokingDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_final = None

    def load_data(self):
        """Loads the dataset and selects the predictors used in the study."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.file_path)
        selected_cols = ['age', 'weight(kg)', 'Gtp', 'HDL', 'gender', 'smoking']
        self.df = self.df[selected_cols].copy()
        print(f"Data loaded. Shape: {self.df.shape}")

    def perform_eda_and_stats(self):
        """Tests each predictor's association with smoking status.

        Mann-Whitney U is used for the numerical features (distributions are
        skewed, so a non-parametric test is appropriate) and Chi-Square for
        gender.
        """
        print("\nStatistical hypothesis testing (alpha = 0.05):")

        numerical_features = ['age', 'weight(kg)', 'Gtp', 'HDL']
        smokers = self.df[self.df['smoking'] == 1]
        non_smokers = self.df[self.df['smoking'] == 0]

        for col in numerical_features:
            stat, p_val = mannwhitneyu(smokers[col], non_smokers[col])
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            print(f"   - {col}: p-value={p_val:.4e} ({sig})")

        contingency = pd.crosstab(self.df['gender'], self.df['smoking'])
        chi2, p_val, _, _ = chi2_contingency(contingency)
        print(f"   - gender: p-value={p_val:.4e} ({'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'})")

    def transform_data(self):
        """Corrects skewness (log1p for GTP/HDL, sqrt for weight) and one-hot
        encodes gender. Returns the modelling-ready DataFrame."""
        print("\nApplying log and sqrt transformations...")
        self.df['Gtp_log'] = np.log1p(self.df['Gtp'])
        self.df['HDL_log'] = np.log1p(self.df['HDL'])
        self.df['weight_sqrt'] = np.sqrt(self.df['weight(kg)'])

        final_cols = ['age', 'gender', 'weight_sqrt', 'Gtp_log', 'HDL_log', 'smoking']
        self.df_final = self.df[final_cols].copy()

        self.df_final = pd.get_dummies(self.df_final, columns=['gender'], drop_first=True)
        print("Data transformation complete.")
        return self.df_final


if __name__ == "__main__":
    processor = SmokingDataProcessor("smoking.csv")
    processor.load_data()
    processor.perform_eda_and_stats()
    df_clean = processor.transform_data()
    print(df_clean.head())
