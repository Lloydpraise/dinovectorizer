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
        initialize_system()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        print("🖼️ Processing incoming image...")
        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        
        # Load as RGBA first to accurately read transparency for color extraction
        image_rgba = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGBA')
        
        # --- 1. DOMINANT COLOR EXTRACTION ---
        print("🎨 Extracting dominant colors...")
        tiny_img = image_rgba.resize((50, 50))
        pixels = tiny_img.load()
        color_counts = {}
        
        for y in range(tiny_img.height):
            for x in range(tiny_img.width):
                r, g, b, a = pixels[x, y]
                
                # Skip transparent pixels
                if a < 128: 
                    continue
                # Skip pure whites, blacks, and grays
                if (r > 240 and g > 240 and b > 240) or (r < 15 and g < 15 and b < 15): 
                    continue 
                
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                color_counts[hex_code] = color_counts.get(hex_code, 0) + 1
                
        # Sort by occurrence and grab top 3
        top_colors = sorted(color_counts, key=color_counts.get, reverse=True)[:3]
        print(f"🎨 Top Colors Found: {top_colors}")

        # --- 2. CENTER 70% CROP & RESIZE ---
        print("✂️ Cropping center 70%...")
        image_rgb = image_rgba.convert('RGB')
        w, h = image_rgb.size
        
        crop_w = int(w * 0.70)
        crop_h = int(h * 0.70)
        crop_x = int((w - crop_w) / 2)
        crop_y = int((h - crop_h) / 2)
        
        image_cropped = image_rgb.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        image_cropped.thumbnail((512, 512))

        # --- 3. VECTORIZATION ---
        print("🧠 Generating vector embedding...")
        inputs = processor(images=image_cropped, return_tensors="pt")
        with torch_module.no_grad():
            outputs = model(**inputs)
        
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        final_vector = vector[:768]
        
        # Print sample to verify vector generation
        print(f"📊 Vector Sample (first 5 dims): {final_vector[:5]}")

        # --- 4. DATABASE MATCHING ---
        print("🔍 Searching Supabase...")
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': final_vector,
            'query_colors': top_colors,
            'match_threshold': 0.35, 
            'match_count': 6
        }).execute()
        
        print(f"🎉 Found {len(response.data)} matches!")
        return jsonify({
            "success": True, 
            "matches": response.data,
            "colors_detected": top_colors
        })

    except Exception as e:
        import traceback
        print(f"🚨 Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)