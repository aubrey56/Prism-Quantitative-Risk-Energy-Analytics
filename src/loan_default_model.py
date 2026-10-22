"""
Retail Banking: Personal Loan Credit Risk & Expected Loss Engine
----------------------------------------------------------------
Trains Logistic Regression & Random Forest classifiers to predict Probability of Default (PD)
and computes regulatory Expected Loss (EL = PD * LGD * EAD).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class LoanDefaultModel:

    def __init__(self, data_path: str, lgd: float = 0.90):
        self.df = pd.read_csv(data_path)
        self.lgd = lgd  # Loss Given Default (90% LGD = 10% Recovery Rate)
        self.model = LogisticRegression(max_iter=1000)

        self._train_and_validate()

    def _train_and_validate(self):
        X = self.df.drop(columns=["customer_id", "default"], errors="ignore")
        y = self.df["default"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit Logistic Regression
        self.model.fit(X_train, y_train)
        lr_probs = self.model.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_probs)

        # Fit Random Forest for comparative evaluation
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

        print(f"Model Calibration Complete.")
        print(f"Logistic Regression AUC: {lr_auc:.6f}")
        print(f"Random Forest AUC:       {rf_auc:.6f}")

    def predict_pd(self, borrower_features: dict) -> float:
        """Predicts Probability of Default (PD) for a new borrower."""
        input_df = pd.DataFrame([borrower_features])
        pd_val = self.model.predict_proba(input_df)[0, 1]
        return float(pd_val)

    def calculate_expected_loss(
        self, borrower_features: dict, loan_amt_outstanding: float
    ) -> dict:
        """
        Calculates Expected Loss: EL = PD * LGD * EAD
        """
        pd_val = self.predict_pd(borrower_features)
        ead = loan_amt_outstanding  # Exposure at Default
        expected_loss = pd_val * self.lgd * ead

        return {
            "Probability of Default (PD)": round(pd_val, 6),
            "Loss Given Default (LGD)": self.lgd,
            "Exposure at Default (EAD)": ead,
            "Expected Loss (EL)": round(expected_loss, 2),
        }


if __name__ == "__main__":
    risk_engine = LoanDefaultModel("Task 3 and 4_Loan_Data.csv")
