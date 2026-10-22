"""
Advanced Risk Methodology: Dynamic Programming FICO Scorecard Quantization
--------------------------------------------------------------------------
Maximizes log-likelihood to partition continuous FICO scores into K optimal,
risk-monotonic categorical rating tiers.
"""

import numpy as np
import pandas as pd


class FICOQuantizer:

    def __init__(
        self, data_path: str, min_fico: int = 300, max_fico: int = 850
    ):
        self.df = pd.read_csv(data_path)
        self.min_fico = min_fico
        self.max_fico = max_fico
        self.num_scores = max_fico - min_fico + 1

        self.sub_n, self.sub_k, self.sub_ll = self._precompute_matrices()

    def _precompute_matrices(self):
        """Precomputes record counts, default counts, and log-likelihood matrices."""
        fico_counts = (
            self.df["fico_score"]
            .value_counts()
            .reindex(range(self.min_fico, self.max_fico + 1), fill_value=0)
            .values
        )
        default_counts = (
            self.df[self.df["default"] == 1]["fico_score"]
            .value_counts()
            .reindex(range(self.min_fico, self.max_fico + 1), fill_value=0)
            .values
        )

        sub_n = np.zeros((self.num_scores, self.num_scores))
        sub_k = np.zeros((self.num_scores, self.num_scores))
        sub_ll = np.zeros((self.num_scores, self.num_scores))

        for i in range(self.num_scores):
            curr_n, curr_k = 0, 0
            for j in range(i, self.num_scores):
                curr_n += fico_counts[j]
                curr_k += default_counts[j]
                sub_n[i, j] = curr_n
                sub_k[i, j] = curr_k

                if curr_n == 0 or curr_k == 0 or curr_k == curr_n:
                    sub_ll[i, j] = 0.0
                else:
                    p = curr_k / curr_n
                    sub_ll[i, j] = curr_k * np.log(p) + (
                        curr_n - curr_k
                    ) * np.log(1.0 - p)

        return sub_n, sub_k, sub_ll

    def optimize_buckets(self, num_buckets: int = 5) -> list[int]:
        """Finds optimal boundary scores maximizing log-likelihood using Dynamic Programming."""
        dp = np.full((num_buckets + 1, self.num_scores), -np.inf)
        parent = np.zeros((num_buckets + 1, self.num_scores), dtype=int)

        # Base Case
        for v in range(self.num_scores):
            dp[1, v] = self.sub_ll[0, v]

        # DP Recurrence
        for j in range(2, num_buckets + 1):
            for v in range(self.num_scores):
                for u in range(v):
                    val = dp[j - 1, u] + self.sub_ll[u + 1, v]
                    if val > dp[j, v]:
                        dp[j, v] = val
                        parent[j, v] = u

        # Backtrack
        boundaries = []
        curr_v = self.num_scores - 1
        for j in range(num_buckets, 1, -1):
            curr_v = parent[j, curr_v]
            boundaries.append(curr_v + self.min_fico)

        boundaries.reverse()
        return [self.min_fico] + boundaries + [self.max_fico]

    def generate_scorecard_table(self, num_buckets: int = 5) -> pd.DataFrame:
        """Generates the final monotonic 5-tier rating scorecard."""
        bounds = self.optimize_buckets(num_buckets)
        records = []

        for b_idx in range(num_buckets):
            low, high = bounds[b_idx], bounds[b_idx + 1]
            slice_df = self.df[
                (self.df["fico_score"] >= low) & (self.df["fico_score"] <= high)
            ]
            n_i = len(slice_df)
            k_i = slice_df["default"].sum()
            pd_i = k_i / n_i if n_i > 0 else 0.0

            records.append(
                {
                    "FICO Range": f"{low} - {high}",
                    "Total Borrowers": n_i,
                    "Defaults": k_i,
                    "PD (%)": f"{pd_i * 100:.2f}%",
                    "_raw_pd": pd_i,
                }
            )

        summary_df = (
            pd.DataFrame(records)
            .sort_values(by="_raw_pd")
            .reset_index(drop=True)
        )
        summary_df.drop(columns=["_raw_pd"], inplace=True)
        summary_df.index = [f"Rating {i+1}" for i in range(num_buckets)]
        summary_df.index.name = "Optimized Rating"

        return summary_df


if __name__ == "__main__":
    quantizer = FICOQuantizer("Task 3 and 4_Loan_Data.csv")
    scorecard = quantizer.generate_scorecard_table(num_buckets=5)
    print(scorecard)
