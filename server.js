// 1. SILENCE THE CPU PROBE (Must be the very first lines)
process.env.ORT_DISABLE_CPU_PROBE = '1'; 
process.env.ONNXRUNTIME_NODE_INSTALL_SKIP = '1';

const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const sharp = require('sharp'); 

// 2. DELAY TRANSFORMERS IMPORT
// We only import transformers once the server is already "Listening"
let pipeline, RawImage, env;

const app = express();
app.use(cors());
app.use(express.json({ limit: '30mb' })); 

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY);
let extractor = null;

// Health Check Route (Test this in your browser: https://your-app.leapcell.dev/health)
app.get('/health', (req, res) => res.send({ status: 'online', ai_ready: !!extractor }));

async function loadAI() {
    console.log("⏱️ [SYSTEM] Waiting 10s for container stabilization...");
    await new Promise(r => setTimeout(r, 10000));

    try {
        console.log("📥 [SYSTEM] Loading AI Library...");
        // Dynamically require so we don't crash the boot process
        const transformers = require('@xenova/transformers');
        pipeline = transformers.pipeline;
        RawImage = transformers.RawImage;
        env = transformers.env;

        // FORCE PURE WASM MODE (The most stable mode)
        env.allowLocalModels = false;
        env.cacheDir = '/tmp/.cache';
        env.backends.onnx.wasm.numThreads = 1;
        env.backends.onnx.wasm.simd = false; // Prevents the cpuinfo crash loop
        env.backends.onnx.wasm.proxy = false;

        console.log("📥 [SYSTEM] Attempting DINOv2 Base (768-dim) load...");
        extractor = await pipeline('image-feature-extraction', 'Xenova/dinov2-base', { 
            quantized: true 
        });
        console.log("✅ [SYSTEM] AI Model Ready.");
    } catch (error) {
        console.error("❌ [SYSTEM] AI Load Failed:", error.message);
    }
}

app.post('/match', async (req, res) => {
    try {
        if (!extractor) return res.status(503).json({ error: "AI is still booting (takes ~30s on first run)" });
        const { image } = req.body;
        
        console.log("📸 [REQUEST] Processing image...");
        const buffer = Buffer.from(image.replace(/^data:image\/\w+;base64,/, ""), 'base64');
        
        // Center 70% Crop
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
            query_colors: ["#000000"], // Color logic can be added back later
            match_threshold: 0.4,
            match_count: 6
        });

        if (dbError) throw dbError;
        res.json({ success: true, matches });
    } catch (err) {
        console.error("🚨 Request Error:", err.message);
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 [SERVER] Listening on port ${PORT}`);
    loadAI(); // Start AI loading AFTER server is listening
});