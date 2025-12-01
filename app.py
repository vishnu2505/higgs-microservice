from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- 1. MODEL ARCHITECTURE ---
class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedNN, self).__init__()

        # Low-level features: 21 inputs
        self.low_level = nn.Sequential(
            nn.Linear(21, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # High-level features: 7 inputs 
        self.high_level = nn.Sequential(
            nn.Linear(7, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Slicing for 28 total features
        low_features = x[:, :21]
        high_features = x[:, 21:] 

        low_out = self.low_level(low_features)
        high_out = self.high_level(high_features)

        combined = torch.cat([low_out, high_out], dim=1)
        output = self.combined(combined)

        return output
import __main__
setattr(__main__, "PhysicsInformedNN", PhysicsInformedNN)   

# --- 2. MODEL LOADING ---
def load_model():
    path = 'higgs_model_cpu.pkl'
    try:
        if os.path.exists(path):
            print(f"Loading model from {path}...")
            # map_location='cpu' is crucial for Render free tier
            model = torch.load(path, map_location=torch.device('cpu'))
            model.eval()
            return model
        else:
            print(f"WARNING: {path} not found. App will crash on prediction.")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

# Full list of 28 features for documentation
FEATURE_NAMES = [
    # Low Level (21)
    "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
    "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag",
    "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag",
    "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b-tag",
    "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
    # High Level (7)
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
]

# --- 3. API ENDPOINTS ---

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "HIGGS Boson Detection Microservice",
        "status": "active",
        "input_features": 28,
        "feature_list": FEATURE_NAMES,
        "endpoints": {
            "/predict": "POST - Send JSON with 'features' list of 28 floats"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' key"}), 400
        
        features = data['features']
        
        # Handle single sample input (convert to list of lists)
        if isinstance(features[0], (int, float)):
            batch = [features]
            is_single = True
        else:
            batch = features
            is_single = False

        # Validate feature length
        for i, sample in enumerate(batch):
            if len(sample) != 28:
                return jsonify({
                    "error": f"Input vector has {len(sample)} features, expected 28."
                }), 400

        # Convert to tensor and predict
        tensor_in = torch.FloatTensor(batch)
        with torch.no_grad():
            outputs = model(tensor_in).numpy().flatten()

        results = []
        for score in outputs:
            prob = float(score)
            results.append({
                "prediction": "signal" if prob > 0.5 else "background",
                "probability": prob,
                "is_higgs": prob > 0.5
            })

        if is_single:
            return jsonify(results[0])
        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to generate a valid test vector for the user
@app.route('/predict/example', methods=['GET'])
def predict_example():
    # Example vector with 28 features
    example_vec = [
        0.869, -0.635, 0.225, 0.327, -0.689, 
        0.754, -0.248, -1.092, 0.000, 0.812, 
        0.213, -1.416, 0.000, 0.902, -0.362, 
        0.219, 1.137, 0.521, -0.857, 1.052, 0.945, # End of low-level (21)
        2.117, 3.467, 0.976, 2.317, 1.687, 3.827, 1.000 # High-level (7)
    ]
    
    # Run prediction locally if possible
    if model:
        with torch.no_grad():
            out = model(torch.FloatTensor([example_vec]))
            prob = float(out.item())
            pred_class = "signal" if prob > 0.5 else "background"
    else:
        prob = 0.0
        pred_class = "model_not_loaded"

    return jsonify({
        "example_input": example_vec,
        "example_output": {
            "prediction": pred_class,
            "probability": prob
        }
    })

if __name__ == '__main__':
    # Render provides the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)