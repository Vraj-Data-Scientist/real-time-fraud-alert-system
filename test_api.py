import requests
import json
import pandas as pd

# API endpoint
url = 'http://3.135.216.208:5000/predict'

# Load dataset to extract realistic samples
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg': 'oldBalanceOrig', 'newbalanceOrig': 'newBalanceOrig',
                        'oldbalanceDest': 'oldBalanceDest', 'newbalanceDest': 'newBalanceDest'})

# Select one fraudulent and one genuine transaction
fraud_sample = df[df['isFraud'] == 1].iloc[0][
    ['step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest']].to_dict()
genuine_sample = df[(df['isFraud'] == 0) & (df['type'].isin(['TRANSFER', 'CASH_OUT']))].iloc[0][
    ['step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest']].to_dict()

# Sample transactions
transactions = [
    fraud_sample,
    genuine_sample,
    # Use the example from the API's root endpoint
    {
        "step": 1,
        "type": "TRANSFER",
        "amount": 100000.00,
        "oldBalanceOrig": 100000.00,
        "newBalanceOrig": 0.00,
        "oldBalanceDest": 0.00,
        "newBalanceDest": 0.00
    }
]

# Test each transaction
for i, transaction in enumerate(transactions, 1):
    try:
        # Send POST request
        response = requests.post(url, json=transaction)

        # Check response status
        if response.status_code == 200:
            result = response.json()
            print(f"\nTransaction {i}:")
            print(f"Input: {json.dumps(transaction, indent=2)}")
            print(f"Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"Is Fraud: {result['is_fraud']}")
            print(f"Alert: {result['alert']}")
        else:
            print(f"\nTransaction {i} failed with status {response.status_code}: {response.json()['error']}")

    except Exception as e:
        print(f"\nTransaction {i} error: {str(e)}")