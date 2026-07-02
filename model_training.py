import os

import matplotlib

matplotlib.use("Agg")  # Render to files; no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = "figures"


class SmokingModelTrainer:
    def __init__(self, df):
        self.df = df
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        os.makedirs(FIGURES_DIR, exist_ok=True)

    def split_data(self):
        """Splits data into Train (70%) and Test (30%), stratified on the target."""
        X = self.df.drop("smoking", axis=1)
        y = self.df["smoking"]
        self.feature_names = list(X.columns)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Scale features (fit on train only to avoid data leakage)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(f"\nData Split: Train={self.X_train.shape}, Test={self.X_test.shape}")

    def train_models(self):
        """Trains KNN, Logistic Regression, and Random Forest."""
        print("\nTraining Models...")

        self.models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
            "Random Forest (n=400)": RandomForestClassifier(
                n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=42
            ),
        }

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            print(f"   - {name} trained.")

    def evaluate_models(self):
        """Evaluates models on the held-out test set and saves ROC curves and
        confusion matrices to the figures/ directory."""
        print("\nModel Evaluation Results (30% held-out test set):")

        roc_fig, roc_ax = plt.subplots(figsize=(8, 6))

        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]

            auc = roc_auc_score(self.y_test, y_prob)
            print(f"\n--- {name} ---")
            print(classification_report(self.y_test, y_pred))
            print(f"AUC Score: {auc:.4f}")

            RocCurveDisplay.from_predictions(
                self.y_test, y_prob, name=f"{name} (AUC={auc:.3f})", ax=roc_ax
            )

            cm_fig, cm_ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                self.y_test, y_pred, display_labels=["Non-smoker", "Smoker"],
                cmap="Blues", ax=cm_ax, colorbar=False,
            )
            cm_ax.set_title(f"Confusion Matrix — {name}")
            cm_slug = name.split(" (")[0].lower().replace(" ", "_")
            cm_fig.tight_layout()
            cm_fig.savefig(os.path.join(FIGURES_DIR, f"confusion_matrix_{cm_slug}.png"), dpi=150)
            plt.close(cm_fig)

        roc_ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        roc_ax.set_title("ROC Curves — Test Set")
        roc_ax.legend(loc="lower right")
        roc_fig.tight_layout()
        roc_fig.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
        plt.close(roc_fig)
        print(f"\nFigures saved to {FIGURES_DIR}/")

    def plot_feature_importance(self):
        """Visualizes and saves Random Forest feature importance."""
        rf = self.models["Random Forest (n=400)"]
        importances = rf.feature_importances_

        fig, ax = plt.subplots(figsize=(9, 5))
        order = importances.argsort()[::-1]
        sns.barplot(
            x=importances[order],
            y=[self.feature_names[i] for i in order],
            hue=[self.feature_names[i] for i in order],
            palette="viridis", legend=False, ax=ax,
        )
        ax.set_title("Random Forest Feature Importance")
        ax.set_xlabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
        plt.close(fig)
        print(f"Feature importance plot saved to {FIGURES_DIR}/feature_importance.png")
