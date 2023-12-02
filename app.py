from flask import Flask, jsonify, request
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
# Load the trained models
model_min_temp = joblib.load('trained_model_min_temp.joblib')
model_max_temp = joblib.load('trained_model_max_temp.joblib')

# Endpoint to predict max temperature
@app.route('/predict/temperatures/max_t', methods=['POST'])
def predict_max_temperature():
    data = request.get_json()
    num_days = int(data['num_days'])

    # Prepare data for prediction
    future_dates = pd.DataFrame({'Days': range(num_days)})

    # Predict max temperatures for future dates
    max_temp_predictions = model_max_temp.predict(future_dates)

    # Prepare response for max temperature predictions
    response = []
    start_date = pd.Timestamp.now().date() + pd.Timedelta(days=1)
    for i in range(num_days):
        date = (start_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        response.append({"date": date, "max_temperature": max_temp_predictions[i]})

    return jsonify({"predictions": response})

# Endpoint to predict min temperature
@app.route('/predict/temperatures/min_t', methods=['POST'])
def predict_min_temperature():
    data = request.get_json()
    num_days = int(data['num_days'])

    # Prepare data for prediction
    future_dates = pd.DataFrame({'Days': range(num_days)})

    # Predict min temperatures for future dates
    min_temp_predictions = model_min_temp.predict(future_dates)

    # Prepare response for min temperature predictions
    response = []
    start_date = pd.Timestamp.now().date() + pd.Timedelta(days=1)
    for i in range(num_days):
        date = (start_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        response.append({"date": date, "min_temperature": min_temp_predictions[i]})

    return jsonify({"predictions": response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

