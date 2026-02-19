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
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# 1. Supabase Setup
print("🔌 Connecting to Supabase...")
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_ANON_KEY")
)
print("✅ Supabase Connected.")

# 2. DINOv2 Loading
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
            print("🚨 Error: No image provided in request")
            return jsonify({"error": "No image"}), 400

        print("🖼️ Processing incoming image...")
        
        # A. Decode Image
        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGB')
        
        # B. 70% Center Crop & Resize
        w, h = image.size
        image = image.crop((w * 0.15, h * 0.15, w * 0.85, h * 0.85))
        image.thumbnail((512, 512))
        print(f"✂️ Image cropped and resized to: {image.size}")

        # C. Vectorize
        print("🧠 Generating vector embedding...")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get CLS token vector
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        final_vector = vector[:768]
        print("✅ Vector generated successfully.")

        # D. Database Match via Supabase RPC
        print("🔍 Searching Supabase for matches...")
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': final_vector,
            'query_colors': ["#000000"], 
            'match_threshold': 0.35,
            'match_count': 6
        }).execute()
        
        print(f"🎉 Found {len(response.data)} matches!")
        return jsonify({"success": True, "matches": response.data})

    except Exception as e:
        print(f"🚨 Error during match: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)