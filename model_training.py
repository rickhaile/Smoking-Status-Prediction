import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class SmokingModelTrainer:
    def __init__(self, df):
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}

    def split_data(self):
        """Splits data into Train (70%) and Test (30%)."""
        X = self.df.drop('smoking', axis=1)
        y = self.df['smoking']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(f"\n✂️ Data Split: Train={self.X_train.shape}, Test={self.X_test.shape}")

    def train_models(self):
        """Trains KNN, Logistic Regression, and Random Forest."""
        print("\n🚀 Training Models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'KNN (k=15)': KNeighborsClassifier(n_neighbors=15),
            'Random Forest (n=400)': RandomForestClassifier(n_estimators=400, class_weight='balanced', n_jobs=-1, random_state=42)
        }

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            print(f"   ✅ {name} Trained.")

    def evaluate_models(self):
        """Evaluates models on Test Set and prints metrics."""
        print("\n🏆 Model Evaluation Results:")
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            auc = roc_auc_score(self.y_test, y_prob)
            print(f"\n--- {name} ---")
            print(classification_report(self.y_test, y_pred))
            print(f"AUC Score: {auc:.4f}")
            
            # Optional: Show confusion matrix
            # cm = confusion_matrix(self.y_test, y_pred)
            # print(cm)

    def plot_feature_importance(self):
        """Visualizes Feature Importance for Random Forest."""
        rf = self.models['Random Forest (n=400)']
        importances = rf.feature_importances_
        feature_names = ['age', 'weight_sqrt', 'Gtp_log', 'HDL_log', 'gender_M'] # Manually aligned after get_dummies
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names, palette='viridis')
        plt.title("Random Forest Feature Importance")
        plt.show()

if __name__ == "__main__":
    # This allows running this file standalone if data is provided
    pass