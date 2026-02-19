// 1. INITIAL CONFIGURATION (Must be at the very top)
const { pipeline, RawImage, env } = require('@xenova/transformers');

// Force the engine to use WebAssembly (WASM) to bypass Leapcell CPU probing errors
env.allowLocalModels = false;
env.cacheDir = '/tmp/.cache';
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false; 

// 2. IMPORTS
const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
const sharp = require('sharp'); 

// 3. APP SETUP
const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' })); // Allow large image uploads

// 4. SUPABASE CONNECTION
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY;
const supabase = (SUPABASE_URL && SUPABASE_KEY) ? createClient(SUPABASE_URL, SUPABASE_KEY) : null;

if (supabase) {
    console.log("✅ [SYSTEM] Supabase client initialized.");
} else {
    console.error("❌ [SYSTEM] Supabase credentials missing from Environment Variables!");
}

// 5. AI MODEL LOADING (Singleton)
let extractor;

async function loadModel() {
    console.log("📥 [SYSTEM] Loading DINOv2 Base (768-dim) into WASM memory...");
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

// 6. ROUTES
app.get('/', (req, res) => res.send('AI Visual Matcher API is Online'));

app.post('/match', async (req, res) => {
    try {
        const { image } = req.body;
        if (!image) return res.status(400).json({ success: false, error: "No image provided" });
        if (!extractor) return res.status(503).json({ success: false, error: "AI Model is still loading..." });

        console.log("📸 [REQUEST] Received image. Processing...");
        
        // Step A: Decode Base64
        const base64Data = image.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');
        
        // Step B: Color Extraction (using a tiny 50x50 version of the image)
        const tinyBuffer = await sharp(buffer).resize(50, 50).raw().toBuffer({ resolveWithObject: true });
        const colorCounts = {};
        const pixels = tinyBuffer.data;
        for (let i = 0; i < pixels.length; i += 3) {
            const r = pixels[i], g = pixels[i+1], b = pixels[i+2];
            // Skip background colors (whites/blacks)
            if ((r > 240 && g > 240 && b > 240) || (r < 15 && g < 15 && b < 15)) continue; 
            const hex = "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
            colorCounts[hex] = (colorCounts[hex] || 0) + 1;
        }
        const topColors = Object.keys(colorCounts).sort((a, b) => colorCounts[b] - colorCounts[a]).slice(0, 3);
        console.log("🎨 [PROCESS] Colors:", topColors);

        // Step C: 70% Center Crop & Resize to 512px
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

        // Step D: AI Vectorization
        console.time("⏱️ AI Inference");
        const rawImage = new RawImage(
            processedImageBuffer.data, 
            processedImageBuffer.info.width, 
            processedImageBuffer.info.height, 
            3 // RGB
        );
        
        const output = await extractor(rawImage, { pooling: 'mean', normalize: true });
        const vector = Array.from(output.data).slice(0, 768);
        console.timeEnd("⏱️ AI Inference");
        console.log(`🤖 [AI] Vectorized to ${vector.length} dimensions.`);

        // Step E: Database Match in Supabase
        if (!supabase) throw new Error("Database connection not established.");
        
        console.time("⏱️ DB Match");
        const { data: matches, error: dbError } = await supabase.rpc('match_products_advanced', {
            query_embedding: vector,
            query_colors: topColors,
            match_threshold: 0.4,
            match_count: 6
        });
        console.timeEnd("⏱️ DB Match");

        if (dbError) throw dbError;

        console.log(`🎉 [SUCCESS] Found ${matches?.length || 0} matches.`);
        res.json({ success: true, matches, colors_detected: topColors });

    } catch (err) {
        console.error("🚨 [ERROR]:", err.message);
        res.status(500).json({ success: false, error: err.message });
    }
});

// 7. START SERVER
const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 [SERVER] Leapcell AI Server running on port ${PORT}`);
});