import os
import io
import base64
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# 1. Initialize Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

# 2. Lazy Global AI Models
processor = None
model = None

def load_ai():
    global processor, model
    if model is None:
        print("📥 Loading DINOv2 Base (768-dim) into PyTorch...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()
        print("✅ AI Model Ready.")

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "API is Online", "model_loaded": model is not None})

@app.route('/match', methods=['POST'])
def match():
    try:
        # Load model on the first request if not loaded
        load_ai()
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        print("📸 Processing image...")
        
        # A. Decode
        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        # B. 70% Crop & Resize
        w, h = image.size
        cropped = image.crop((w*0.15, h*0.15, w*0.85, h*0.85))
        cropped.thumbnail((512, 512))
        
        # C. Vectorize
        inputs = processor(images=cropped, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        
        # D. Match
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': vector[:768],
            'query_colors': ["#000000"], 
            'match_threshold': 0.4,
            'match_count': 6
        }).execute()
        
        return jsonify({"success": True, "matches": response.data})
        
    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start AI loading 5 seconds after server boot to allow environment to settle
    time.sleep(5)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)