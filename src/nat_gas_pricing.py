"""
Commodities Desk: Natural Gas Forward Curve & Storage Valuation Engine
---------------------------------------------------------------------
Implements harmonic regression to model price trends + seasonality,
and provides a contract valuation function for physical storage facilities.
"""

import numpy as np
import pandas as pd


class NaturalGasPricer:

    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.df["Dates"] = pd.to_datetime(self.df["Dates"])
        self.df = self.df.sort_values("Dates").reset_index(drop=True)

        self.start_date = self.df["Dates"].min()
        self.df["Days"] = (self.df["Dates"] - self.start_date).dt.days

        self.coefs = None
        self._fit_forward_curve()

    def _fit_forward_curve(self):
        """Fits linear trend + annual harmonic seasonality via OLS."""
        X = self.df["Days"].values
        y = self.df["Prices"].values

        sines = np.sin(2 * np.pi * X / 365.25)
        cosines = np.cos(2 * np.pi * X / 365.25)
        A = np.vstack([np.ones_like(X), X, sines, cosines]).T

        self.coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    def get_price(self, date_str: str) -> float:
        """Predicts gas spot price for any target date."""
        target_date = pd.to_datetime(date_str)
        days = (target_date - self.start_date).days

        sine = np.sin(2 * np.pi * days / 365.25)
        cosine = np.cos(2 * np.pi * days / 365.25)

        return (
            self.coefs[0]
            + self.coefs[1] * days
            + self.coefs[2] * sine
            + self.coefs[3] * cosine
        )

    def valuate_storage_contract(
        self,
        injections: list[tuple[str, float]],
        withdrawals: list[tuple[str, float]],
        monthly_lease_rate: float = 100000.0,
        processing_fee_per_unit: float = 0.01,
        max_capacity: float = 1500000.0,
    ) -> dict:
        """Calculates Net Fair Value of a physical gas storage contract."""
        total_injected = sum(v for _, v in injections)
        total_withdrawn = sum(v for _, v in withdrawals)

        if total_injected > max_capacity:
            raise ValueError(
                f"Injected volume ({total_injected}) exceeds max facility capacity ({max_capacity})."
            )

        # 1. Purchase Costs
        purchase_cost = sum(
            vol * self.get_price(d) for d, vol in injections
        )

        # 2. Sales Revenue
        sales_revenue = sum(
            vol * self.get_price(d) for d, vol in withdrawals
        )

        # 3. Processing Surcharges
        processing_cost = (
            total_injected + total_withdrawn
        ) * processing_fee_per_unit

        # 4. Lease Cost (fractional monthly duration)
        all_dates = [pd.to_datetime(d) for d, _ in injections + withdrawals]
        duration_days = (max(all_dates) - min(all_dates)).days
        lease_cost = monthly_lease_rate * (duration_days / 30.4375)

        # Net Fair Value
        net_value = (
            sales_revenue - purchase_cost - processing_cost - lease_cost
        )

        return {
            "Gross Revenue": round(sales_revenue, 2),
            "Purchase Cost": round(purchase_cost, 2),
            "Commodity Spread": round(sales_revenue - purchase_cost, 2),
            "Processing Fees": round(processing_cost, 2),
            "Facility Lease": round(lease_cost, 2),
            "Net Fair Value": round(net_value, 2),
        }


if __name__ == "__main__":
    pricer = NaturalGasPricer("Task 3 and 4_Loan_Data.csv")  # Adapt path
    print("Forward Curve Coefs:", pricer.coefs)
