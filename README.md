# Fraud Detection in Financial Payment Services

## Overview
This project develops a machine learning model to detect fraudulent transactions in financial payment services using a dataset of over 6 million transactions. The dataset is highly imbalanced, with only ~0.3% of transactions being fraudulent. The primary model, an **XGBoost Classifier**, leverages feature engineering and class weighting to achieve high performance, evaluated using the Area Under the Precision-Recall Curve (AUPRC). The project includes exploratory data analysis (EDA), data cleaning, feature engineering, and visualizations to uncover fraud patterns.

## Dataset
The dataset (`PS_20174392719_1491204439457_log.csv`) contains 6,362,620 financial transactions, with 8,213 confirmed fraudulent cases (0.129% fraud rate). It is sourced from a simulated or anonymized financial payment service and includes:

- **Features**:
  - `step`: Time step (assumed hours, up to 743).
  - `type`: Transaction type (TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT).
  - `amount`: Transaction amount.
  - `nameOrig`: Originator account ID.
  - `oldBalanceOrig`, `newBalanceOrig`: Originator balance before and after the transaction.
  - `nameDest`: Destination account ID.
  - `oldBalanceDest`, `newBalanceDest`: Destination balance before and after the transaction.
  - `isFraud`: Binary target (1 = fraud, 0 = non-fraud).
  - `isFlaggedFraud`: Binary flag for suspected fraud (unreliable, set only 16 times).
- **Key Observations**:
  - Fraud occurs only in `TRANSFER` (4,097 cases) and `CASH_OUT` (4,116 cases) transactions.
  - Zero destination balances (`oldBalanceDest` and `newBalanceDest`) with non-zero `amount` are strong fraud indicators (~49.56% of fraud vs. ~0.06% of non-fraud).
  - `isFlaggedFraud` is inconsistent and dropped from modeling.
  - Account IDs (`nameOrig`, `nameDest`) and merchant roles are unreliable.

**Note**: The dataset is not included in the repository due to its size (~1 GB). Download it from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place `PS_20174392719_1491204439457_log.csv` in the `data/` folder.

## Model Information
- **Algorithm**: XGBoost Classifier
  - **Hyperparameters**:
    - `max_depth=3`
    - `scale_pos_weight`: Ratio of non-fraud to fraud instances (~337:1) to handle imbalance.
    - `n_jobs=4` for parallel processing.
  - **Training**:
    - Data split: 80% training, 20% test (`random_state=5`).
    - Features: `type` (binary: TRANSFER=0, CASH_OUT=1), `amount`, `oldBalanceOrig`, `newBalanceOrig`, `oldBalanceDest`, `newBalanceDest`, `errorBalanceOrig`, `errorBalanceDest`.
    - Engineered Features:
      - `errorBalanceOrig`: `newBalanceOrig + amount - oldBalanceOrig`
      - `errorBalanceDest`: `oldBalanceDest + amount - newBalanceDest`
    - Dropped Features: `nameOrig`, `nameDest`, `isFlaggedFraud`.
    - Data Cleaning:
      - Zero destination balances with non-zero `amount` set to -1.
      - Zero originator balances with non-zero `amount` set to NaN.
  - **Metric**: AUPRC, preferred over AUROC due to class imbalance.

## Performance Results
The XGBoost Classifier was evaluated on the test set (1,272,524 samples, including ~1,643 fraud cases) using AUPRC, with additional insights from feature importance and learning curves.

- **AUPRC**: **0.9926**, indicating excellent performance in detecting fraud despite severe imbalance.
- **Feature Importance** (based on `cover` metric):
  1. `errorBalanceDest`: Most critical, capturing destination balance anomalies.
  2. `errorBalanceOrig`: Originator balance anomalies.
  3. `amount`: Transaction amount patterns.
  4. `type`: TRANSFER vs. CASH_OUT.
  5. `oldBalanceDest`, `newBalanceDest`, `oldBalanceOrig`, `newBalanceOrig`: Balance-related features.
  6. `step`: Marginally important (transaction time).
- **Learning Curves**:
  - Show a slightly underfit model, with high but not fully converged training and cross-validation AUPRC scores.
  - Suggests potential for improvement with more data or hyperparameter tuning.
- **Key Observations**:
  - The model effectively handles the ~0.3% fraud rate using class weighting, avoiding data loss from undersampling.
  - Engineered error features (`errorBalanceDest`, `errorBalanceOrig`) are highly discriminative.
  - Zero destination balances are a strong fraud indicator, captured by the -1 replacement.
  - The model generalizes well, but subtle fraud patterns (e.g., lack of expected TRANSFER-to-CASH_OUT chains) require careful feature engineering.

## Installation
To run the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vraj-Data-Scientist/fraud-detection-model-for-transaction-data.git
   cd fraud-detection-model-for-transaction-data
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   ```
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn
   xgboost
   joblib
   ```

4. **Download the Dataset**:
   - Download `PS_20174392719_1491204439457_log.csv` from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).
   - Place it in the `data/` folder.

## Usage
1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook predicting-fraud-in-financial-payment-services.ipynb
   ```

2. **Run the Notebook**:
   - Execute cells sequentially to perform EDA, data cleaning, feature engineering, model training, and evaluation.
   - Visualizations (e.g., strip plots, 3D scatter plots, heatmaps, feature importance) are generated inline.
   - The trained model is saved as `fraud_model.pkl`.

3. **Load and Use the Saved Model**:
   ```python
   import joblib
   model = joblib.load('fraud_model.pkl')
   predictions = model.predict(test_data)
   ```


## Visualizations
- **Temporal Distribution**: Fraudulent transactions are evenly distributed over time, unlike genuine transactions, which show periodic patterns.
- **Amount Distribution**: Fraudulent transactions cluster in a narrower, lower amount range compared to genuine transactions.
- **Error-Based Features**: `errorBalanceDest` and `errorBalanceOrig` clearly separate fraud from non-fraud in 2D and 3D plots.
- **Correlation Heatmaps**: Fraudulent and genuine transactions show distinct correlation patterns, confirming discriminative features.
- **Feature Importance**: `errorBalanceDest` and `errorBalanceOrig` dominate model predictions.
- **Learning Curves**: Indicate slight underfitting, with potential for improved performance.



---
