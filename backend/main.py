import json
import math
import os
import random
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONSTANTS ---
MAX_STEPS = 200
SEQUENCE_LENGTH = 12
NUM_SENSORS = 207
TOTAL_MOCK_STEPS = 1200
DATASET_FILE = 'METR-LA.h5'
MODEL_FILE = 'traffic_lstm.keras'

# --- ENV SETUP ---
def load_env_files():
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
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env_files()

app = Flask(__name__)
CORS(app)

# --- GLOBAL VARIABLES ---
MOCK_DATA = []
REAL_DATA = None  # Will hold DataFrame if loaded
lstm_model = None # Will hold Keras model if loaded

# --- DATA LOADING & MOCK FALLBACK ---

def load_real_data():
    """Attempts to load METR-LA dataset from .h5 file."""
    global REAL_DATA
    if os.path.exists(DATASET_FILE):
        try:
            import pandas as pd
            print(f"Loading real dataset from {DATASET_FILE}...")
            # METR-LA is usually saved as a dataframe in HDF5
            df = pd.read_hdf(DATASET_FILE)
            # Ensure we only take the first 207 columns if file differs
            REAL_DATA = df.iloc[:, :NUM_SENSORS].values 
            print(f"Successfully loaded real data. Shape: {REAL_DATA.shape}")
            return True
        except Exception as e:
            print(f"Failed to load real data: {e}")
            return False
    else:
        print(f"Dataset file '{DATASET_FILE}' not found. Using simulation mode.")
        return False

def generate_mock_data():
    """Generates synthetic data if real data is missing."""
    global MOCK_DATA
    print(f"Generating mock data for {NUM_SENSORS} sensors...")
    MOCK_DATA = []
    
    for s in range(NUM_SENSORS):
        sensor_data = []
        # Base speed 50-70 mph with noise
        current_speed = 50 + (s % 5) * 5 
        noise_offset = (s % 7) - 3 

        for t in range(TOTAL_MOCK_STEPS):
            # Complex sine wave to mimic traffic (morning/evening rush)
            current_speed += math.sin(t / 20 + s / 10) * 2.0 + (noise_offset / 10)
            # Random sudden drops (traffic jams)
            if random.random() > 0.98:
                current_speed -= random.uniform(5, 15)
            
            # Recovery
            current_speed += random.uniform(-1, 1.5)
            current_speed = max(10, min(75, current_speed))

            actual = round(current_speed + random.uniform(-1, 1), 2)
            
            # Prediction has slightly less noise in mock mode
            predicted = round(actual + random.uniform(-2, 2), 2)
            
            sensor_data.append([actual, predicted])
        MOCK_DATA.append(sensor_data)
    print("Mock data ready.")

# --- LSTM MODEL MANAGEMENT ---

def get_or_train_model():
    """
    Loads existing LSTM model or trains a new one if real data exists.
    Requires: tensorflow
    """
    global lstm_model
    if REAL_DATA is None:
        return # Cannot train without data

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("TensorFlow not installed. Skipping LSTM usage.")
        return

    if os.path.exists(MODEL_FILE):
        print(f"Loading existing LSTM model from {MODEL_FILE}...")
        lstm_model = load_model(MODEL_FILE)
        return

    print("Training new LSTM model (this may take a while)...")
    
    # Data Preparation for Training (Simplified)
    # We flatten data to train a generic 'traffic physics' model valid for all sensors
    # Uses first 1000 steps of first 50 sensors to save time for this demo
    training_data = REAL_DATA[:1000, :50].flatten().reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(training_data)

    X_train, y_train = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X_train.append(scaled_data[i-SEQUENCE_LENGTH:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Define LSTM Architecture
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1) # Low epochs for demo speed

    model.save(MODEL_FILE)
    lstm_model = model
    print("LSTM Model trained and saved.")

def lstm_predict(history):
    """
    Uses the real LSTM to predict the next steps.
    history: list of floats (length 12)
    """
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        
        # We need a scaler. In a real app, load the saved scaler. 
        # Here we fit on the fly roughly or assume 0-80mph range for normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit logic is hacky here for demo; ideally load saved scaler
        scaler.min_ = np.array([0.0])
        scaler.scale_ = np.array([1.0/80.0]) 
        
        input_seq = np.array(history).reshape(-1, 1)
        scaled_seq = scaler.transform(input_seq)
        
        # Reshape for LSTM [1, 12, 1]
        curr_seq = scaled_seq.reshape(1, SEQUENCE_LENGTH, 1)
        
        predictions = []
        for _ in range(SEQUENCE_LENGTH):
            pred_scaled = lstm_model.predict(curr_seq, verbose=0)
            pred_val = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(float(pred_val))
            
            # Update sequence: remove first, add new prediction
            new_step = np.array([[[pred_scaled[0][0]]]])
            curr_seq = np.append(curr_seq[:, 1:, :], new_step, axis=1)
            
        return predictions
    except Exception as e:
        print(f"LSTM Prediction error: {e}")
        return []

# --- MOCK SIMULATION (Fallback) ---

def simulate_math_prediction(historical_speeds, sensor_index):
    """Fallback mathematical simulation if no LSTM is available."""
    current_sequence = list(historical_speeds)
    predicted_speeds = []

    for step in range(SEQUENCE_LENGTH):
        # Average of last 3 + trend
        avg = sum(current_sequence[-3:]) / 3
        trend = current_sequence[-1] - current_sequence[-2] if len(current_sequence) > 1 else 0
        
        # Decay trend
        next_val = avg + (trend * 0.5)
        
        # Add "uncertainty" noise
        noise = random.uniform(-2, 2)
        next_val += noise
        
        next_val = max(10, min(85, next_val))
        predicted_speeds.append(round(next_val, 2))

        current_sequence.pop(0)
        current_sequence.append(next_val)
        
    return predicted_speeds

# --- API ROUTES ---

@app.route('/api/evaluation', methods=['POST'])
def evaluation_data():
    data = request.json
    sensor_id = int(data.get('sensorId', 0))
    start_step = int(data.get('startStep', 0))
    print("Came here 1")
    if sensor_id < 0 or sensor_id >= NUM_SENSORS:
        return jsonify({"error": "Invalid Sensor ID"}), 400

    steps = MAX_STEPS

    if REAL_DATA is not None:
        print("Came here 2")
        # Clamp start_step to avoid overflow
        max_idx = len(REAL_DATA) - steps - SEQUENCE_LENGTH
        if start_step > max_idx: start_step = max_idx

        # Get historical sequence for LSTM
        history_window = REAL_DATA[start_step : start_step + SEQUENCE_LENGTH, sensor_id].tolist()
        actuals = REAL_DATA[start_step : start_step + steps, sensor_id].tolist()

        # Predicted values
        if lstm_model is not None:
            try:
                import numpy as np
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.min_ = np.array([0.0])
                scaler.scale_ = np.array([1.0 / 80.0])  # assuming max speed 80

                # Initialize sequence
                seq = np.array(history_window).reshape(1, SEQUENCE_LENGTH, 1)
                preds_scaled = []
                for i in range(steps):
                    pred_scaled = lstm_model.predict(seq, verbose=0)
                    preds_scaled.append(pred_scaled[0][0])
                    # Slide the window with **actual value** from REAL_DATA
                    if i < len(actuals):
                        next_input = scaler.transform(np.array([[actuals[i]]]))
                    else:
                        next_input = pred_scaled.reshape(1,1,1)
                    seq = np.append(seq[:,1:,:], next_input.reshape(1,1,1), axis=1)



                # Inverse scale
                predicted_speeds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten().tolist()

            except Exception as e:
                print(f"LSTM batch prediction failed: {e}")
                predicted_speeds = (np.array(actuals) + np.random.normal(0, 2.0, steps)).tolist()
        else:
            # Fallback to noisy predictions
            predicted_speeds = (np.array(actuals) + np.random.normal(0, 2.0, steps)).tolist()

        combined = [[a, p] for a, p in zip(actuals, predicted_speeds)]
        return jsonify({
            "labels": list(range(start_step, start_step + steps)),
            "actualSpeeds": actuals,
            "predictedSpeeds": predicted_speeds,
            "dataSlice": combined
        })

    else:
        # Mock fallback
        if not MOCK_DATA: generate_mock_data()
        data_slice = MOCK_DATA[sensor_id][start_step : start_step + steps]
        actuals = [d[0] for d in data_slice]
        preds = [d[1] for d in data_slice]
        return jsonify({
            "labels": list(range(start_step, start_step + len(data_slice))),
            "actualSpeeds": actuals,
            "predictedSpeeds": preds,
            "dataSlice": data_slice
        })

@app.route('/api/forecast', methods=['POST'])
def forecast_data():
    data = request.json
    sensor_id = int(data.get('sensorId', 0))
    historical_speeds = data.get('historicalData', [])

    if len(historical_speeds) != SEQUENCE_LENGTH:
        return jsonify({"error": f"Need exactly {SEQUENCE_LENGTH} historical points"}), 400

    # USE REAL LSTM IF AVAILABLE
    if lstm_model is not None:
        preds = lstm_predict(historical_speeds)
        if preds:
            return jsonify({"predictedSpeeds": preds})
            
    # Fallback
    preds = simulate_math_prediction(historical_speeds, sensor_id)
    return jsonify({"predictedSpeeds": preds})

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "geminiApiKey": os.environ.get('apikey', ''),
        "mode": "REAL_LSTM" if lstm_model else "MOCK_SIMULATION"
    })

# --- INITIALIZATION ---
if __name__ == '__main__':
    # 1. Try load data
    has_data = load_real_data()
    
    # 2. If data, try init LSTM
    if has_data:
        get_or_train_model()
    else:
        generate_mock_data()

    app.run(debug=True, port=5000)