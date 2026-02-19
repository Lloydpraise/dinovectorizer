import os
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

# 1. Initialize Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

# 2. Load the AI Model natively in PyTorch
print("📥 Loading DINOv2 Base (768-dim) into PyTorch...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()
print("✅ AI Model Ready.")

def extract_colors(image, num_colors=3):
    tiny = image.copy()
    tiny.thumbnail((50, 50))
    colors = tiny.getcolors(2500)
    if not colors: return ["#000000"]
    
    top_colors = []
    colors.sort(key=lambda x: x[0], reverse=True)
    for count, pixel in colors:
        r, g, b = pixel[:3]
        if (r > 240 and g > 240 and b > 240) or (r < 15 and g < 15 and b < 15): continue
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        top_colors.append(hex_color)
        if len(top_colors) >= num_colors: break
    
    return top_colors if top_colors else ["#000000"]

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "API is Online", "model_ready": True})

@app.route('/match', methods=['POST'])
def match():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
            
        print("📸 Processing new image...")
        
        # A. Decode Base64
        base64_str = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # B. Extract Colors
        top_colors = extract_colors(image)
        print(f"🎨 Colors: {top_colors}")
        
        # C. Crop 70% Center & Resize
        w, h = image.size
        left, top = w * 0.15, h * 0.15
        right, bottom = w * 0.85, h * 0.85
        cropped = image.crop((left, top, right, bottom))
        cropped.thumbnail((512, 512))
        
        # D. Vectorize with PyTorch
        inputs = processor(images=cropped, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        vector = vector[:768] 
        print(f"🤖 Vectorized to {len(vector)} dims.")
        
        # E. Database Match
        if not supabase:
            raise Exception("Supabase credentials missing.")
            
        response = supabase.rpc('match_products_advanced', {
            'query_embedding': vector,
            'query_colors': top_colors,
            'match_threshold': 0.4,
            'match_count': 6
        }).execute()
        
        print(f"🎉 Found {len(response.data)} matches.")
        return jsonify({"success": True, "matches": response.data, "colors_detected": top_colors})
        
    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)