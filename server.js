// 1. SILENCE THE CPU PROBE (Must be the very first lines)
process.env.ONNXRUNTIME_NODE_INSTALL_SKIP = '1';
process.env.ORT_LOGGING_LEVEL = '3'; // Only show critical errors

const { pipeline, RawImage, env } = require('@xenova/transformers');

// 2. FORCE THE ENGINE TO BE "DUMB" & STABLE
try {
    env.allowLocalModels = false;
    env.cacheDir = '/tmp/.cache';
    
    // Force pure WebAssembly (WASM) to stop native C++ hardware probing
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.proxy = false;
    
    // Disable the advanced CPU features that cause the crash loop
    env.backends.onnx.logLevel = 'error';
} catch (e) {
    console.error("Warning: Could not set AI environment flags:", e.message);
}

const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const sharp = require('sharp'); 

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' })); 

const supabase = (process.env.SUPABASE_URL && process.env.SUPABASE_ANON_KEY) 
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY) 
    : null;

let extractor = null;

// 3. LAZY LOAD THE MODEL (Wait 2 seconds after boot to let the system stabilize)
async function loadModel() {
    console.log("⏱️ [SYSTEM] Waiting for environment stabilization...");
    await new Promise(r => setTimeout(r, 2000));
    
    console.log("📥 [SYSTEM] Loading DINOv2 Base (768-dim) into WASM...");
    try {
        extractor = await pipeline('image-feature-extraction', 'Xenova/dinov2-base', { 
            quantized: true 
        });
        console.log("✅ [SYSTEM] AI Model Ready.");
    } catch (error) {
        console.error("❌ [SYSTEM] Failed to load AI model:", error.message);
    }
}
loadModel();

app.get('/', (req, res) => res.send('API is Online'));

app.post('/match', async (req, res) => {
    try {
        const { image } = req.body;
        if (!image) return res.status(400).json({ error: "No image" });
        if (!extractor) return res.status(503).json({ error: "AI still loading" });

        console.log("📸 [REQUEST] Received image...");
        const base64Data = image.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');
        
        // Color Extraction
        const tinyBuffer = await sharp(buffer).resize(50, 50).raw().toBuffer({ resolveWithObject: true });
        const colorCounts = {};
        const pixels = tinyBuffer.data;
        for (let i = 0; i < pixels.length; i += 3) {
            const r = pixels[i], g = pixels[i+1], b = pixels[i+2];
            if ((r > 240 && g > 240 && b > 240) || (r < 15 && g < 15 && b < 15)) continue; 
            const hex = "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
            colorCounts[hex] = (colorCounts[hex] || 0) + 1;
        }
        const topColors = Object.keys(colorCounts).sort((a, b) => colorCounts[b] - colorCounts[a]).slice(0, 3);

        // Center 70% Crop
        const metadata = await sharp(buffer).metadata();
        const processedImageBuffer = await sharp(buffer)
            .extract({ 
                left: Math.round(metadata.width * 0.15), 
                top: Math.round(metadata.height * 0.15), 
                width: Math.round(metadata.width * 0.7), 
                height: Math.round(metadata.height * 0.7) 
            })
            .resize(512, 512, { fit: 'inside' })
            .raw()
            .toBuffer({ resolveWithObject: true });

        // Vectorize
        const rawImage = new RawImage(processedImageBuffer.data, processedImageBuffer.info.width, processedImageBuffer.info.height, 3);
        const output = await extractor(rawImage, { pooling: 'mean', normalize: true });
        const vector = Array.from(output.data).slice(0, 768);

        // DB Query
        const { data: matches, error: dbError } = await supabase.rpc('match_products_advanced', {
            query_embedding: vector,
            query_colors: topColors,
            match_threshold: 0.4,
            match_count: 6
        });

        if (dbError) throw dbError;
        res.json({ success: true, matches, colors_detected: topColors });

    } catch (err) {
        console.error("🚨 Request Error:", err.message);
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', () => console.log(`🚀 Server on port ${PORT}`));