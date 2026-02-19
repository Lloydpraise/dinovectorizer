import os
# Force Hugging Face to use the writable /tmp directory on Leapcell
os.environ['HF_HOME'] = '/tmp/huggingface'

import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

app = Flask(__name__)
CORS(app)

print("📥 Loading DINOv2 (PyTorch CPU version)...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()
print("✅ AI Model Ready.")

@app.route('/')
def health():
    return "API is Online and DINOv2 is loaded."

@app.route('/match', methods=['POST'])
def match():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        # A. Decode & Center 70% Crop
        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        w, h = image.size
        image = image.crop((w * 0.15, h * 0.15, w * 0.85, h * 0.85))
        image.thumbnail((512, 512))

        # B. Vectorize
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # C. Get CLS token vector (standard for DINOv2)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()

        return jsonify({"success": True, "vector": vector[:768]})

    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)