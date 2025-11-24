import json
import math
import os
import random
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONSTANTS (Must match frontend) ---
MAX_STEPS = 200
SEQUENCE_LENGTH = 12
NUM_SENSORS = 207
TOTAL_MOCK_STEPS = 1200 # Increased buffer to allow startStep up to 1000 + 200 steps


def load_env_files():
    """
    Minimal .env loader so we don't need external dependencies.
    Supports KEY=VALUE lines; ignores comments and quoted values.
    """
    possible_paths = [
        Path(__file__).resolve().parent.parent / '.env',
        Path(__file__).resolve().parent / '.env',
    ]

    for env_path in possible_paths:
        if not env_path.exists():
            continue
        with env_path.open('r', encoding='utf-8') as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


load_env_files()

app = Flask(__name__)
# Enable CORS for frontend development (allowing JavaScript to call the API)
CORS(app)

MOCK_DATA = []

def generate_mock_data():
    """Generates synthetic data simulating Actual vs. Predicted traffic speeds."""
    global MOCK_DATA
    
    print(f"Generating mock data for {NUM_SENSORS} sensors over {TOTAL_MOCK_STEPS} steps...")
    
    for s in range(NUM_SENSORS):
        sensor_data = []
        current_speed = 50 + (s % 5) * 5 
        noise = (s % 7) - 3 

        for t in range(TOTAL_MOCK_STEPS):
            # Simulate cyclical speed pattern (daily/weekly cycles)
            current_speed += math.sin(t / 10 + s / 10) * 1.5 + noise / 10
            current_speed = max(20, min(70, current_speed))

            # Simulate actual measurement with some noise
            actual_speed = round(current_speed + random.uniform(-2.5, 2.5), 2)
            
            # Simulate prediction error (prediction is usually close to actual)
            prediction_error = random.uniform(-2.5, 2.5)
            if t > 0 and abs(sensor_data[-1][0] - actual_speed) < 3:
                prediction_error *= 0.7 

            predicted_speed = round(actual_speed + prediction_error, 2)
            predicted_speed = max(20, min(70, predicted_speed))

            sensor_data.append([actual_speed, predicted_speed])
        MOCK_DATA.append(sensor_data)
    
    print("Mock data generation complete.")

def simulate_multi_step_prediction(historical_speeds, sensor_index):
    """
    Simulates the LSTM model's recursive 12-step prediction for the Forecast tab.
    """
    current_sequence = list(historical_speeds)
    predicted_speeds = []

    for step in range(SEQUENCE_LENGTH):
        # Simplified simulation: Prediction based on the average of the last 4 inputs
        last_four_avg = sum(current_sequence[-4:]) / 4
        
        # Introduce increasing uncertainty as the forecast horizon extends
        uncertainty_factor = (SEQUENCE_LENGTH - step) / SEQUENCE_LENGTH
        variation = random.uniform(-2.5, 2.5) * 5 * uncertainty_factor
        
        predicted_speed = last_four_avg + variation
        predicted_speed += (sensor_index / NUM_SENSORS) * 1.5 
        predicted_speed = max(20.0, min(90.0, predicted_speed))
        
        predicted_speeds.append(round(predicted_speed, 2))

        # Update the sequence for the next recursive prediction
        current_sequence.pop(0) 
        current_sequence.append(predicted_speed) 
        
    return predicted_speeds

@app.route('/api/evaluation', methods=['POST'])
def evaluation_data():
    """Endpoint for serving the long-term Actual vs. Predicted data for Evaluation mode."""
    data = request.json
    sensor_id = int(data.get('sensorId', 0))
    start_step = int(data.get('startStep', 0))

    # Input Validation checks
    if not MOCK_DATA or sensor_id < 0 or sensor_id >= NUM_SENSORS:
        return jsonify({"error": "Invalid Sensor ID or data not initialized."}), 400
    
    # Crucial check: Ensure start_step is within the valid range
    max_valid_start_step = len(MOCK_DATA[0]) - MAX_STEPS
    if start_step < 0 or start_step > max_valid_start_step:
        return jsonify({"error": f"Invalid Start Step. Must be between 0 and {max_valid_start_step}."}), 400

    # Get the slice of data for the chart plot and table
    sensor_data = MOCK_DATA[sensor_id]
    data_slice = sensor_data[start_step : start_step + MAX_STEPS]

    # Structure the response
    actual_speeds = [d[0] for d in data_slice]
    predicted_speeds = [d[1] for d in data_slice]
    labels = list(range(start_step, start_step + len(data_slice)))

    return jsonify({
        "labels": labels,
        "actualSpeeds": actual_speeds,
        "predictedSpeeds": predicted_speeds,
        "dataSlice": data_slice
    })

@app.route('/api/forecast', methods=['POST'])
def forecast_data():
    """Endpoint for generating the 12-step forecast based on manual input."""
    data = request.json
    sensor_id = int(data.get('sensorId', 0))
    historical_speeds = data.get('historicalData', [])

    if len(historical_speeds) != SEQUENCE_LENGTH:
        return jsonify({"error": f"Historical data must contain exactly {SEQUENCE_LENGTH} values."}), 400

    if sensor_id < 0 or sensor_id >= NUM_SENSORS:
        return jsonify({"error": "Invalid Sensor ID."}), 400

    # Run the 12-step recursive simulation
    predicted_speeds = simulate_multi_step_prediction(historical_speeds, sensor_id)

    return jsonify({
        "predictedSpeeds": predicted_speeds
    })

@app.route('/api/config', methods=['GET'])
def get_frontend_config():
    """Expose limited configuration so the frontend can fetch secrets from .env."""
    api_key = os.environ.get('apikey')
    if not api_key:
        return jsonify({"error": "Gemini API key not configured on the server."}), 500
    return jsonify({"geminiApiKey": api_key})

# Initialize mock data when the application starts
with app.app_context():
    generate_mock_data()

if __name__ == '__main__':
    # To run: python app.py
    # This will serve the API at http://127.0.0.1:5000
    app.run(debug=True)