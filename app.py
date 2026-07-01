from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
try:
    clf = joblib.load(os.path.join(BASE_DIR, 'stage1_classifier.pkl'))
    scaler1 = joblib.load(os.path.join(BASE_DIR, 'scaler_stage1.pkl'))
    reg = joblib.load(os.path.join(BASE_DIR, 'stage2_regressor.pkl'))
    scaler2 = joblib.load(os.path.join(BASE_DIR, 'scaler_stage2.pkl'))
    print("✅ Models loaded successfully!")
    print(f"Classifier expects {scaler1.n_features_in_} features")
    print(f"Regressor expects {scaler2.n_features_in_} features")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    clf = scaler1 = reg = scaler2 = None

def calculate_dewpoint(temp_c, humidity):
    """Calculate dew point temperature"""
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point

def estimate_precipitation(humidity, pressure, temp):
    """Simple precipitation likelihood (0 or 1)"""
    if humidity > 85 and pressure < 1005 and temp > 0:
        return 1
    return 0

def preprocess_input(data):
    temp_c = float(data.get('temperature', 16.8))
    humidity = float(data.get('humidity', 68))
    wind_ms = float(data.get('wind_speed', 7.2))
    pressure_hpa = float(data.get('pressure', 1008))
    hour = int(data.get('hour', 14))
    month = int(data.get('month', 2))
    
    dew_point = calculate_dewpoint(temp_c, humidity)
    temp_dewpoint_spread = temp_c - dew_point
    precip_binary = estimate_precipitation(humidity, pressure_hpa, temp_c)
    
    features = np.array([[
        temp_c, humidity, wind_ms, pressure_hpa,
        hour, month,
        temp_dewpoint_spread, precip_binary
    ]])
    
    return features

@app.route('/')
def serve_ui():
    return send_from_directory(BASE_DIR, 'interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if clf is None or reg is None:
            return jsonify({'success': False, 'error': 'Models not loaded'}), 500
            
        data = request.json
        features = preprocess_input(data)
        
        # Classification
        features_scaled = scaler1.transform(features)
        is_reduced = clf.predict(features_scaled)[0]
        confidence = clf.predict_proba(features_scaled)[0].max()
        
        # Regression
        features_scaled2 = scaler2.transform(features)
        visibility_km = reg.predict(features_scaled2)[0]
        visibility_km = max(0.1, min(25.0, visibility_km))
        
        return jsonify({
            'success': True,
            'visibility_km': round(visibility_km, 1),
            'visibility_miles': round(visibility_km / 1.60934, 1),
            'confidence': int(confidence * 100),
            'is_reduced': bool(is_reduced),
            'condition': 'Reduced visibility' if is_reduced else 'Clear conditions'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': clf is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)