// 1. HARD-SILENCE THE ENGINE (Must be the very first lines)
process.env.ORT_DISABLE_CPU_PROBE = '1'; 
process.env.ONNXRUNTIME_NODE_INSTALL_SKIP = '1';

const { pipeline, RawImage, env } = require('@xenova/transformers');

// 2. ULTRA-STABLE AI CONFIGURATION
env.allowLocalModels = false;
env.cacheDir = '/tmp/.cache'; // Leapcell only allows writes in /tmp
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.simd = false; // Disable SIMD to prevent instruction probing
env.backends.onnx.wasm.proxy = false;

const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const sharp = require('sharp'); 

const app = express();
app.use(cors());
app.use(express.json({ limit: '30mb' })); 

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY);

let extractor = null;

// Helper to log RAM usage
const logMemory = () => {
    const used = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log(`📊 [MEMORY] Current Usage: ${Math.round(used)} MB`);
};

// Check if the environment is Windows and adjust cache directory accordingly
if (process.platform === 'win32') {
    env.cacheDir = 'C:\\temp'; // Change to a writable directory on Windows
}

// 3. LAZY LOAD WITH STABILIZATION
async function loadModel() {
    console.log("⏱️ [SYSTEM] Stabilization delay (3s)...");
    await new Promise(r => setTimeout(r, 3000));
    
    console.log("📥 [SYSTEM] Attempting DINOv2 Base (768-dim) load...");
    logMemory();

    try {
        extractor = await pipeline('image-feature-extraction', 'Xenova/dinov2-base', { 
            quantized: true 
        });
        console.log("✅ [SYSTEM] AI Model Ready.");
        logMemory();
    } catch (error) {
        console.error("❌ [SYSTEM] Failed to load AI model:", error.message);
    }
}
loadModel();

app.get('/', (req, res) => {
    logMemory();
    res.send('AI Visual Matcher is Active');
});

app.post('/match', async (req, res) => {
    try {
        const { image } = req.body;
        if (!image) return res.status(400).json({ error: "No image" });
        if (!extractor) return res.status(503).json({ error: "Model loading" });

        console.log("📸 [REQUEST] Processing new image...");
        const buffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), 'base64');
        
        // Color Extraction
        const tiny = await sharp(buffer).resize(50, 50).raw().toBuffer({ resolveWithObject: true });
        const colorCounts = {};
        for (let i = 0; i < tiny.data.length; i += 3) {
            const r = tiny.data[i], g = tiny.data[i+1], b = tiny.data[i+2];
            if ((r > 240 && g > 240 && b > 240) || (r < 15 && g < 15 && b < 15)) continue; 
            const hex = "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
            colorCounts[hex] = (colorCounts[hex] || 0) + 1;
        }
        const topColors = Object.keys(colorCounts).sort((a, b) => colorCounts[b] - colorCounts[a]).slice(0, 3);

        // Center 70% Crop & Resize
        const meta = await sharp(buffer).metadata();
        const processed = await sharp(buffer)
            .extract({ 
                left: Math.round(meta.width * 0.15), 
                top: Math.round(meta.height * 0.15), 
                width: Math.round(meta.width * 0.7), 
                height: Math.round(meta.height * 0.7) 
            })
            .resize(512, 512, { fit: 'inside' })
            .raw()
            .toBuffer({ resolveWithObject: true });

        // Vectorize
        const raw = new RawImage(processed.data, processed.info.width, processed.info.height, 3);
        const output = await extractor(raw, { pooling: 'mean', normalize: true });
        const vector = Array.from(output.data).slice(0, 768);

        // Supabase Match
        const { data: matches, error: dbError } = await supabase.rpc('match_products_advanced', {
            query_embedding: vector,
            query_colors: topColors,
            match_threshold: 0.4,
            match_count: 6
        });

        if (dbError) throw dbError;
        console.log(`🎉 [SUCCESS] Found ${matches?.length || 0} matches.`);
        res.json({ success: true, matches, colors_detected: topColors });

    } catch (err) {
        console.error("🚨 Request Error:", err.message);
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', () => console.log(`🚀 Server listening on port ${PORT}`));