from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
from flask_cors import CORS
from flasgger import Swagger

app = Flask(__name__)
CORS(app)

# Initialize Swagger
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

template = {
    "swagger": "2.0",
    "info": {
        "title": "HIGGS Boson Detection API",
        "description": "API for classifying particle collision events using a Physics-Informed Neural Network.",
        "version": "1.0"
    }
}

swagger = Swagger(app, config=swagger_config, template=template)

# --- 1. MODEL ARCHITECTURE & FIX ---
class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedNN, self).__init__()
        self.low_level = nn.Sequential(
            nn.Linear(21, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.high_level = nn.Sequential(
            nn.Linear(7, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(128 + 32, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
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
            model = torch.load(path, map_location=torch.device('cpu'))
            model.eval()
            return model
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

model = load_model()

FEATURE_NAMES = [
    "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
    "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag",
    "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag",
    "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b-tag",
    "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
]

# --- 3. ENDPOINTS ---

@app.route('/', methods=['GET'])
def home():
    """
    API Information
    ---
    responses:
      200:
        description: Returns service metadata
    """
    return jsonify({
        "service": "HIGGS Boson Detection Microservice",
        "status": "active",
        "input_features": 28,
        "docs_url": "/docs"
    })

@app.route('/health', methods=['GET'])
def health():
    """
    Health Check
    ---
    responses:
      200:
        description: Check if service is live
    """
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/features', methods=['GET'])
def features():
    """
    Get Feature Names
    ---
    responses:
      200:
        description: Returns list of the 28 required input features
        examples:
          application/json: {"count": 28, "names": ["lepton_pT", "..."]}
    """
    return jsonify({"count": 28, "names": FEATURE_NAMES})

@app.route('/predict/example', methods=['GET'])
def predict_example():
    """
    Get Example Data
    ---
    description: Returns a valid sample input vector and its prediction. Useful for testing.
    responses:
      200:
        description: A sample JSON object for the /predict endpoint
    """
    example_vec = [
        0.869, -0.635, 0.225, 0.327, -0.689, 0.754, -0.248, -1.092, 
        0.000, 0.812, 0.213, -1.416, 0.000, 0.902, -0.362, 0.219, 
        1.137, 0.521, -0.857, 1.052, 0.945, 2.117, 3.467, 0.976, 
        2.317, 1.687, 3.827, 1.000
    ]
    
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
        "prediction": pred_class,
        "probability": prob
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a Prediction
    ---
    parameters:
      - name: body
        in: body
        required: true
        description: Input features (28 floats)
        schema:
          type: object
          properties:
            features:
              type: array
              items:
                type: number
              example: [0.869, -0.635, 0.225, 0.327, -0.689, 0.754, -0.248, -1.092, 0.0, 0.812, 0.213, -1.416, 0.0, 0.902, -0.362, 0.219, 1.137, 0.521, -0.857, 1.052, 0.945, 2.117, 3.467, 0.976, 2.317, 1.687, 3.827, 1.0]
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: string
            probability:
              type: number
            is_higgs:
              type: boolean
    """
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' key"}), 400
        
        features = data['features']
        
        if isinstance(features[0], (int, float)):
            batch = [features]
            is_single = True
        else:
            batch = features
            is_single = False

        for i, sample in enumerate(batch):
            if len(sample) != 28:
                return jsonify({"error": f"Sample {i} has {len(sample)} features, expected 28."}), 400

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)