from ml.config import ModelConfig
from ml.feature_extractor import load_clip, compute_image_features
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import joblib
import json
import os
from io import BytesIO
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app = Flask(__name__)

# Global model cache - load once on startup


class ModelCache:
    def __init__(self):
        self.probe = None
        self.text_features = None
        self.clip_model = None
        self.preprocess = None
        self.config = None
        self.classes = None

    def load_models(self):
        """Load all models on startup"""
        artifacts_dir = '/home/nikhil/Documents/dev/python/projs/SIH/ml/artifacts'

        print("Loading runtime config...")
        with open(os.path.join(artifacts_dir, 'runtime_config.json')) as f:
            self.config = json.load(f)

        self.classes = self.config['classes']
        print(f"Classes: {self.classes}")

        print("Loading trained probe...")
        self.probe = joblib.load(os.path.join(
            artifacts_dir, 'linear_probe.joblib'))

        print("Loading cached text features...")
        self.text_features = torch.tensor(
            np.load(os.path.join(artifacts_dir, 'text_features.npy'))
        )

        print("Loading CLIP model...")
        model_cfg = ModelConfig()
        self.clip_model, self.preprocess, _ = load_clip(model_cfg)

        print("✅ All models loaded successfully!")


# Initialize model cache
cache = ModelCache()


def softmax(logits):
    """Compute softmax with numerical stability"""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / (exp.sum(axis=-1, keepdims=True) + 1e-9)


def predict_image(image_pil):
    """Run inference on a PIL image"""
    # Preprocess image
    image_tensor = cache.preprocess(image_pil).unsqueeze(0)

    # Get image features
    img_features = compute_image_features(
        cache.clip_model, image_tensor, 'cuda')

    # Zero-shot classification
    zs_logits = (img_features @ cache.text_features.to('cuda').t()) / 0.01
    zs_probs = softmax(zs_logits.detach().cpu().numpy())

    # Reduce prompts to classes (6 prompts per class)
    num_classes = len(cache.classes)
    prompts_per_class = 6

    zs_probs_class = []
    for c_idx in range(num_classes):
        start = c_idx * prompts_per_class
        end = (c_idx + 1) * prompts_per_class
        zs_probs_class.append(zs_probs[:, start:end].max(axis=1))

    zs_probs_class = np.stack(zs_probs_class, axis=1)

    # Linear probe classification
    probe_probs = cache.probe.predict_proba(img_features.cpu().numpy())

    # Blend predictions (70% probe, 30% zero-shot)
    alpha = 0.7
    blended = alpha * probe_probs + (1 - alpha) * zs_probs_class

    # Get final prediction
    pred_idx = int(blended[0].argmax())
    confidence = float(blended[0].max())

    # Create probability distribution
    all_probs = {cache.classes[i]: float(
        blended[0][i]) for i in range(len(cache.classes))}

    return {
        'predicted_class': cache.classes[pred_idx],
        'confidence': confidence,
        'all_probabilities': all_probs
    }


@app.route('/')
def home():
    return jsonify({
        'message': 'Civic Issue Classification API',
        'status': 'ready',
        'classes': cache.classes,
        'endpoints': {
            '/predict': 'POST - Upload image for classification',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models_loaded': True})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Load and process image
        image = Image.open(BytesIO(file.read())).convert('RGB')

        # Run prediction
        result = predict_image(image)

        return jsonify({
            'success': True,
            'result': result,
            'message': f'Image classified as {result["predicted_class"]} with {result["confidence"]:.1%} confidence'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting Civic Issue Classification API...")
    print("Loading models...")
    cache.load_models()
    print("🌟 Server ready! Visit http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
