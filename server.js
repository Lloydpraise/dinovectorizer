const express = require('express');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');
// IMPORT 'env' to configure the AI engine
const { pipeline, RawImage, env } = require('@xenova/transformers');
const sharp = require('sharp'); 

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' })); 

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_ANON_KEY;
let supabase;

if (SUPABASE_URL && SUPABASE_KEY) {
    supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
    console.log("✅ Supabase client initialized.");
} else {
    console.warn("⚠️ Warning: Supabase environment variables are missing.");
}

// --- CONTAINER FIXES ---
env.allowLocalModels = false; 
env.cacheDir = '/tmp/.cache'; // Force downloads into the writable temp folder

let extractor;

async function loadModel() {
    console.log("📥 Downloading and Loading DINOv2 Base (768-dim) into RAM...");
    try {
        extractor = await pipeline('image-feature-extraction', 'Xenova/dinov2-base', { 
            quantized: true,
            // Stop ONNX from probing the restricted CPU files
            session_options: {
                intra_op_num_threads: 1,
                inter_op_num_threads: 1
            }
        });
        console.log("✅ AI Model Ready.");
    } catch (error) {
        console.error("❌ Failed to load AI model:", error);
    }
}
loadModel();

app.get('/', (req, res) => {
    res.send('AI Visual Matcher is running.');
});

app.post('/match', async (req, res) => {
    try {
        const { image } = req.body;
        if (!image) return res.status(400).json({ success: false, error: "No image base64 provided in request body." });
        if (!extractor) return res.status(503).json({ success: false, error: "AI Model is still booting up. Try again in a few seconds." });

        console.log("📸 Received image payload. Starting processing...");
        
        // 1. Decode Image Buffer
        const base64Data = image.replace(/^data:image\/\w+;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');
        
        // 2. Extract Colors
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
        console.log("🎨 Dominant Colors:", topColors);

        // 3. Golden Rule: Crop Center 70% & Resize
        const metadata = await sharp(buffer).metadata();
        const cropW = Math.round(metadata.width * 0.7);
        const cropH = Math.round(metadata.height * 0.7);
        const left = Math.round((metadata.width - cropW) / 2);
        const top = Math.round((metadata.height - cropH) / 2);

        const processedImageBuffer = await sharp(buffer)
            .extract({ left, top, width: cropW, height: cropH })
            .resize(512, 512, { fit: 'inside' })
            .raw()
            .toBuffer({ resolveWithObject: true });
            
        console.log(`✂️ Cropped & Resized to: ${processedImageBuffer.info.width}x${processedImageBuffer.info.height}`);

        // 4. AI Vectorization
        console.time("⏱️ AI Inference");
        const rawImage = new RawImage(
            processedImageBuffer.data, 
            processedImageBuffer.info.width, 
            processedImageBuffer.info.height, 
            3
        );
        
        const output = await extractor(rawImage, { pooling: 'mean', normalize: true });
        const vector = Array.from(output.data).slice(0, 768);
        console.timeEnd("⏱️ AI Inference");
        console.log(`🤖 Vectorized locally to ${vector.length} dimensions.`);

        // 5. Query Supabase
        if (!supabase) throw new Error("Supabase is not configured on the server.");
        
        console.time("⏱️ DB Match");
        const { data: matches, error: dbError } = await supabase.rpc('match_products_advanced', {
            query_embedding: vector,
            query_colors: topColors,
            match_threshold: 0.4,
            match_count: 6
        });
        console.timeEnd("⏱️ DB Match");

        if (dbError) throw dbError;

        console.log(`🎉 Found ${matches?.length || 0} matches.`);
        res.json({ success: true, matches, colors_detected: topColors });

    } catch (err) {
        console.error("🚨 Server Error:", err.message);
        res.status(500).json({ success: false, error: err.message });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Leapcell AI Server running on port ${PORT}`);
});