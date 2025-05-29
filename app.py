from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('fraud_model.pkl')
except FileNotFoundError:
    raise Exception("Model file 'fraud_model.pkl' not found. Please train and save the model first.")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Fraud Detection API',
        'endpoint': '/predict',
        'method': 'POST',
        'expected_input': {
            'step': 'int',
            'type': 'str (TRANSFER or CASH_OUT)',
            'amount': 'float',
            'oldBalanceOrig': 'float',
            'newBalanceOrig': 'float',
            'oldBalanceDest': 'float',
            'newBalanceDest': 'float'
        },
        'example': {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 100000.00,
            'oldBalanceOrig': 100000.00,
            'newBalanceOrig': 0.00,
            'oldBalanceDest': 0.00,
            'newBalanceDest': 0.00
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate required fields
        required_fields = ['step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Create DataFrame from input
        input_data = pd.DataFrame([data])

        # Preprocess input (mimic notebook)
        # Encode 'type'
        if input_data['type'][0] not in ['TRANSFER', 'CASH_OUT']:
            return jsonify({'error': 'Invalid transaction type. Must be TRANSFER or CASH_OUT'}), 400
        input_data.loc[input_data.type == 'TRANSFER', 'type'] = 0
        input_data.loc[input_data.type == 'CASH_OUT', 'type'] = 1
        input_data['type'] = input_data['type'].astype(int)

        # Handle zero balances
        input_data.loc[(input_data.oldBalanceDest == 0) & (input_data.newBalanceDest == 0) & (input_data.amount != 0),
                       ['oldBalanceDest', 'newBalanceDest']] = -1
        input_data.loc[(input_data.oldBalanceOrig == 0) & (input_data.newBalanceOrig == 0) & (input_data.amount != 0),
                       ['oldBalanceOrig', 'newBalanceOrig']] = np.nan

        # Feature engineering
        input_data['errorBalanceOrig'] = input_data['newBalanceOrig'] + input_data['amount'] - input_data['oldBalanceOrig']
        input_data['errorBalanceDest'] = input_data['oldBalanceDest'] + input_data['amount'] - input_data['newBalanceDest']

        # Select features for prediction
        features = ['step', 'type', 'amount', 'oldBalanceOrig', 'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest',
                    'errorBalanceOrig', 'errorBalanceDest']
        X = input_data[features]

        # Make prediction
        prob = model.predict_proba(X)[:, 1][0]  # Probability of fraud
        prediction = 1 if prob >= 0.5 else 0    # Binary prediction

        # Return result
        response = {
            'fraud_probability': float(prob),
            'is_fraud': int(prediction),
            'alert': 'Fraudulent transaction detected!' if prediction == 1 else 'Transaction appears genuine.'
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)