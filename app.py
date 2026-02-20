import os
os.environ['HF_HOME'] = '/tmp/huggingface'

import io
import base64
import gc
import requests
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, pipeline

app = Flask(__name__)
CORS(app)

supabase = None

def init_db():
    global supabase
    if supabase is None:
        from supabase import create_client
        supabase = create_client(
            os.environ.get("SUPABASE_URL"), 
            os.environ.get("SUPABASE_ANON_KEY")
        )

# --- SHARED HELPER: CLAHE LIGHTING CORRECTION ---
def apply_clahe(image_rgb):
    # OPTIMIZATION: Shrink image slightly before CLAHE to save CPU
    image_rgb.thumbnail((1000, 1000))
    img_np = np.array(image_rgb)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    limg = cv2.merge((cl, a, b))
    final_img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(final_img_np)

# --- SHARED HELPER: SMART CROP ---
def smart_crop(image_rgb, detector):
    image_rgb.thumbnail((800, 800))
    
    detections = detector(image_rgb)
    target_box = None
    if detections:
        valid_detections = [d for d in detections if d['score'] > 0.5]
        if valid_detections:
            target_box = max(valid_detections, key=lambda d: (d['box']['xmax'] - d['box']['xmin']) * (d['box']['ymax'] - d['box']['ymin']))['box']

    if target_box:
        pad_w = (target_box['xmax'] - target_box['xmin']) * 0.1
        pad_h = (target_box['ymax'] - target_box['ymin']) * 0.1
        crop_x = max(0, target_box['xmin'] - pad_w)
        crop_y = max(0, target_box['ymin'] - pad_h)
        crop_w = min(image_rgb.width, target_box['xmax'] + pad_w)
        crop_h = min(image_rgb.height, target_box['ymax'] + pad_h)
        img_cropped = image_rgb.crop((int(crop_x), int(crop_y), int(crop_w), int(crop_h)))
    else:
        w, h = image_rgb.size
        crop_w, crop_h = int(w * 0.80), int(h * 0.80)
        crop_x, crop_y = int((w - crop_w) / 2), int((h - crop_h) / 2)
        img_cropped = image_rgb.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        
    img_cropped.thumbnail((512, 512))
    img_byte_arr = io.BytesIO()
    img_cropped.save(img_byte_arr, format='JPEG', quality=90)
    img_byte_arr.seek(0)
    return Image.open(img_byte_arr)

@app.route('/')
def health():
    return "API is Online. Sequential loading optimized."

@app.route('/match', methods=['POST'])
def match():
    try:
        init_db()
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_rgba = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert('RGBA')
        image_rgb = image_rgba.convert('RGB')
        
        print("💡 Applying CLAHE Lighting Correction...")
        image_rgb = apply_clahe(image_rgb)
        
        # Color Extraction from the light-corrected image
        tiny_img = image_rgb.resize((50, 50))
        pixels = tiny_img.load()
        color_counts = {}
        for y in range(tiny_img.height):
            for x in range(tiny_img.width):
                r, g, b = pixels[x, y]
                if (r > 240 and g > 240 and b > 240) or (r < 15 and g < 15 and b < 15): continue 
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                color_counts[hex_code] = color_counts.get(hex_code, 0) + 1
        top_colors = sorted(color_counts, key=color_counts.get, reverse=True)[:3]

        print("📥 Loading DETR...")
        detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        final_image_for_ai = smart_crop(image_rgb, detector)
        del detector
        gc.collect()

        print("📥 Loading DINOv2...")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()

        inputs = processor(images=final_image_for_ai, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        final_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768]

        del processor, model
        gc.collect()

        response = supabase.rpc('match_products_advanced', {
            'query_embedding': final_vector,
            'query_colors': top_colors,
            'match_threshold': 0.20, 
            'match_count': 6
        }).execute()
        
        return jsonify({"success": True, "matches": response.data, "colors_detected": top_colors})

    except Exception as e:
        import traceback
        print(f"🚨 Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/vectorize', methods=['POST'])
def vectorize_product():
    try:
        init_db()
        data = request.get_json()
        product_id = data.get('id')
        image_urls = data.get('images', [])
        if not product_id or not image_urls:
            return jsonify({"error": "Missing id/images"}), 400

        image_urls = image_urls[:3]
        raw_images = []
        for url in image_urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content)).convert('RGB')
                    raw_images.append(img)
                else: raw_images.append(None)
            except: raw_images.append(None)

        detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        cropped_images = [smart_crop(img, detector) if img else None for img in raw_images]
        del detector
        gc.collect()

        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()

        vectors = []
        for img in cropped_images:
            if img:
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                vectors.append(outputs.last_hidden_state[:, 0, :].squeeze().tolist()[:768])
            else: vectors.append(None)

        del processor, model
        gc.collect()

        update_payload = {'vectorized': True}
        for i, v in enumerate(vectors):
            if v: update_payload[f'vector_{i+1}'] = v

        supabase.table('products').update(update_payload).eq('id', product_id).execute()
        return jsonify({"success": True, "updated_columns": list(update_payload.keys())})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)