import os
# Force Hugging Face to use the writable /tmp directory on Leapcell
os.environ['HF_HOME'] = '/tmp/huggingface'

import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Global placeholders
processor = None
model = None
supabase = None
torch_module = None

def initialize_system():
    """Ultra-lazy load: delays all heavy imports until the first search"""
    global processor, model, supabase, torch_module
    
    if model is None:
        print("⏳ First request: Importing heavy AI libraries...")
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from supabase import create_client
        
        torch_module = torch

        print("🔌 Connecting to Supabase...")
        supabase = create_client(
            os.environ.get("SUPABASE_URL"), 
            os.environ.get("SUPABASE_ANON_KEY")
        )

        print("📥 Loading DINOv2 (This takes a moment)...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()
        print("✅ System Fully Initialized.")

@app.route('/')
def health():
    return "API is Online. Model will load on the first match request."

@app.route('/match', methods=['POST'])
def match():
    try:
        # Trigger the delayed loading
        initialize_system()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        print("🖼️ Processing incoming image...")
        
        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        w, h = image.size
        image = image.crop((w * 0.15, h * 0.15, w * 0.85, h * 0.85))
        image.thumbnail((512, 512))

        print("🧠 Generating vector embedding...")
        inputs = processor(images=image, return_tensors="pt")
        with torch_module.no_grad():
            outputs = model(**inputs)
        
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        final_vector = vector[:768]

        print("🔍 Searching Supabase...")
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': final_vector,
            'query_colors': ["#000000"], 
            'match_threshold': 0.35,
            'match_count': 6
        }).execute()
        
        print(f"🎉 Found {len(response.data)} matches!")
        return jsonify({"success": True, "matches": response.data})

    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)