import os
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
detector = None

def initialize_system():
    global processor, model, supabase, torch_module, detector
    
    if model is None:
        print("⏳ First request: Importing heavy AI libraries...")
        import torch
        from transformers import AutoImageProcessor, AutoModel, pipeline
        from supabase import create_client
        
        torch_module = torch

        print("🔌 Connecting to Supabase...")
        supabase = create_client(
            os.environ.get("SUPABASE_URL"), 
            os.environ.get("SUPABASE_ANON_KEY")
        )

        print("📥 Loading DINOv2 (Feature Extractor)...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()

        print("📥 Loading DETR (Smart Object Cropper)...")
        detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        
        print("✅ System Fully Initialized.")

@app.route('/')
def health():
    return "API is Online. Models will load on the first match request."

@app.route('/match', methods=['POST'])
def match():
    try:
        initialize_system()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        print("🖼️ Processing incoming image...")
        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        
        image_rgba = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGBA')
        image_rgb = image_rgba.convert('RGB')
        
        # --- 1. DOMINANT COLOR EXTRACTION ---
        print("🎨 Extracting dominant colors...")
        tiny_img = image_rgba.resize((50, 50))
        pixels = tiny_img.load()
        color_counts = {}
        
        for y in range(tiny_img.height):
            for x in range(tiny_img.width):
                r, g, b, a = pixels[x, y]
                if a < 128: continue
                if (r > 240 and g > 240 and b > 240) or (r < 15 and g < 15 and b < 15): continue 
                
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                color_counts[hex_code] = color_counts.get(hex_code, 0) + 1
                
        top_colors = sorted(color_counts, key=color_counts.get, reverse=True)[:3]
        print(f"🎨 Top Colors: {top_colors}")

        # --- 2. SMART AUTO-CROP (DETR) ---
        print("🔎 Scanning image for products (Smart Crop)...")
        detections = detector(image_rgb)
        
        target_box = None
        if detections:
            # Filter out weak detections and find the largest object
            valid_detections = [d for d in detections if d['score'] > 0.5]
            if valid_detections:
                target_box = max(valid_detections, key=lambda d: (d['box']['xmax'] - d['box']['xmin']) * (d['box']['ymax'] - d['box']['ymin']))['box']

        if target_box:
            print(f"🎯 Found object! Cropping to bounding box...")
            # Add 10% padding to match JavaScript logic
            pad_w = (target_box['xmax'] - target_box['xmin']) * 0.1
            pad_h = (target_box['ymax'] - target_box['ymin']) * 0.1
            crop_x = max(0, target_box['xmin'] - pad_w)
            crop_y = max(0, target_box['ymin'] - pad_h)
            crop_w = min(image_rgb.width, target_box['xmax'] + pad_w)
            crop_h = min(image_rgb.height, target_box['ymax'] + pad_h)
            image_cropped = image_rgb.crop((int(crop_x), int(crop_y), int(crop_w), int(crop_h)))
        else:
            print("⚠️ No clear object found, falling back to 80% center crop...")
            w, h = image_rgb.size
            crop_w, crop_h = int(w * 0.80), int(h * 0.80)
            crop_x, crop_y = int((w - crop_w) / 2), int((h - crop_h) / 2)
            image_cropped = image_rgb.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            
        image_cropped.thumbnail((512, 512))

        # Replicate JS JPEG compression quality
        img_byte_arr = io.BytesIO()
        image_cropped.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        final_image_for_ai = Image.open(img_byte_arr)

        # --- 3. VECTORIZATION ---
        print("🧠 Generating CLS vector embedding...")
        inputs = processor(images=final_image_for_ai, return_tensors="pt")
        with torch_module.no_grad():
            outputs = model(**inputs)
        
        vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        final_vector = vector[:768]

        # --- 4. DATABASE MATCHING ---
        print("🔍 Searching Supabase...")
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': final_vector,
            'query_colors': top_colors,
            'match_threshold': 0.20, # Raised back up since the smart crop fixes the math
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