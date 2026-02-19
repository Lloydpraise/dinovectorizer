import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# 1. Supabase Setup
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_ANON_KEY"))

# 2. Pure PyTorch Loading (CPU Only)
print("📥 Loading DINOv2 (PyTorch CPU version)...")
processor = ViTImageProcessor.from_pretrained('facebook/dinov2-base')
model = ViTModel.from_pretrained('facebook/dinov2-base')
model.eval()
print("✅ AI Model Ready.")

@app.route('/')
def health():
    return "API is Online"

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
        # Crop to center 70%
        image = image.crop((w * 0.15, h * 0.15, w * 0.85, h * 0.85))
        image.thumbnail((512, 512))

        # B. Vectorize
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # DINOv2 base uses 768 dimensions
        vector = outputs.pooler_output.squeeze().tolist()

        # C. Database Match
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': vector[:768],
            'query_colors': ["#000000"], 
            'match_threshold': 0.35,
            'match_count': 6
        }).execute()

        return jsonify({"success": True, "matches": response.data})

    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)