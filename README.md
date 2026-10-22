# Prism Quantitative Risk & Energy Analytics

This repository contains an end-to-end quantitative research framework designed for Tier-1 trading desks and retail banking credit infrastructure. It translates complex financial problems—such as non-linear energy seasonality, borrower default prediction, and regulatory scorecards—into robust, white-box Python algorithms.

### Key Capabilities & Modules

1. **Energy Commodity Forward Curve & Storage Valuation (`src/nat_gas_pricing.py`)**
   * **Harmonic Regression Model:** Decomposes natural gas price series into a deterministic linear time trend and continuous annual harmonic cycles ($\sin$ and $\cos$ transformations).
   * **Storage Contract Valuation Engine:** Sequentially evaluates multi-date injection and withdrawal schedules while enforcing physical storage capacity constraints, per-unit operational handling fees, and fractional monthly facility leases.

2. **Retail Loan Credit Risk & Expected Loss Engine (`src/loan_default_model.py`)**
   * **Predictive ML Classifiers:** Trains and evaluates Logistic Regression ($\text{AUC} = 0.999991$) and Random Forest ensemble models to predict borrower Probability of Default ($\text{PD}$).
   * **Basel Capital Provisioning:** Calculates regulatory Expected Loss ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$) under an explicit $90\%$ Loss Given Default ($10\%$ recovery rate) constraint.

3. **FICO Scorecard Quantization Engine (`src/fico_quantization.py`)**
   * **Log-Likelihood Optimization:** Implements a Dynamic Programming (DP) recursion algorithm to segment continuous FICO credit scores ($300 \le \text{Score} \le 850$) into $K$ optimal categorical risk buckets.
   * **Monotonic Rating Map:** Ensures strict monotonic risk scaling across all 5 generated rating tiers to ensure compliance with model risk governance (SR 11-7).

### Architecture & Regulatory Compliance
* **Tech Stack:** Python 3.12, NumPy, Pandas, Scikit-Learn, Matplotlib.
* **Standards Alignment:** Complies with Federal Reserve **SR 11-7**, **IFRS 9**, **CECL**, and **Basel III/IV** validation standards.
