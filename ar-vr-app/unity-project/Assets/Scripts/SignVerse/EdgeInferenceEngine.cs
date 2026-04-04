using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

namespace SignVerse
{
    /// <summary>
    /// EdgeInferenceEngine — Orchestrates 64-frame inference on AR Glasses.
    /// Integrates Motion Intelligence, Tokenization, and the "Decomposed Lite" ONNX model.
    /// </summary>
    public class EdgeInferenceEngine : MonoBehaviour
    {
        [Header("Model Configuration")]
        public string modelPath = "Assets/Models/optimized_foundation_model_best_secure.bin";
        public string keyPath = "Assets/Models/model_key.base64";
        public string vocabPath = "Assets/Models/gesture_vocabulary.json";

        private InferenceSession _session;
        private GestureTokenizer _tokenizer;
        private EdgeMotionIntelligence _motionIntelligence;
        private SecureModelLoader _secureLoader;

        private int[] _slidingWindow = new int[64];
        private int _currentIndex = 0;

        public string currentTranslation    = "";
        public float  confidence            = 0.0f;

        // Telemetry accessed by HybridRouter
        public float  lastInferenceLatencyMs = 0f;
        public int[]  lastTokenWindow        = new int[64];
        public int    distinctTokenCount     = 0;
        public float[] lastMotionVector      = new float[848];


        private void Awake()
        {
            _tokenizer = new GestureTokenizer();
            _tokenizer.LoadVocabulary(vocabPath);
            _motionIntelligence = new EdgeMotionIntelligence();
            _secureLoader = new SecureModelLoader();

            // Load ONNX Session (Supporting SECURE .bin or standard .onnx)
            try {
                if (modelPath.EndsWith(".bin")) {
                    byte[] key = _secureLoader.LoadKey(keyPath);
                    byte[] modelBytes = _secureLoader.DecryptModel(modelPath, key);
                    _session = new InferenceSession(modelBytes);
                    Debug.Log("[SignVerse] Secure ONNX Inference Session initialized (from memory).");
                } else {
                    _session = new InferenceSession(modelPath);
                    Debug.Log("[SignVerse] Standard ONNX Inference Session initialized.");
                }
            } catch (System.Exception ex) {
                Debug.LogError($"[SignVerse] Failed to load ONNX model: {ex.Message}");
            }
        }

        public void ProcessUpdate(XRHand leftHand, XRHand rightHand, Transform head, Transform lShoulder, Transform rShoulder)
        {
            if (!_tokenizer.IsLoaded || _session == null) return;

            // 1. Extract dimensions (848 dims)
            float[] features = _motionIntelligence.ProcessFrame(leftHand, rightHand, head, lShoulder, rShoulder);

            // 2. Tokenize (Landmarks -> Token ID)
            int tokenId = _tokenizer.Tokenize(features);

            // 3. Update Sliding Window (Circular Buffer)
            _slidingWindow[_currentIndex] = tokenId;
            _currentIndex = (_currentIndex + 1) % 64;

            // 4. Run Edge Inference (64 frames)
            RunInference();
        }

        private void RunInference()
        {
            // Prepare inputs
            long[] batchDimensions = { 1, 64 };
            DenseTensor<int> tokensTensor = new DenseTensor<int>(batchDimensions);
            
            // Re-order sliding window for the model (sequential)
            for (int i = 0; i < 64; i++)
            {
                int idx = (_currentIndex + i) % 64;
                tokensTensor[0, i] = _slidingWindow[idx];
            }

            // Create causal mask (matching the exporter's static mask)
            bool[] maskData = new bool[64 * 64];
            for (int r = 0; r < 64; r++)
            {
                for (int c = 0; c < 64; c++)
                {
                    maskData[r * 64 + c] = (c > r); // Upper triangular logic
                }
            }
            DenseTensor<bool> maskTensor = new DenseTensor<bool>(maskData, new int[] { 64, 64 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("tokens", tokensTensor),
                NamedOnnxValue.CreateFromTensor("mask", maskTensor)
            };

            using (var results = _session.Run(inputs))
            {
                // Process logits (batch, seq_len, vocab_size)
                var logits = results.First(r => r.Name == "logits").AsTensor<float>();
                
                // For simplicity, take the last token's distribution
                // (1, 64, 512) -> we care about (1, 63, :) to predict token 64
                int bestToken = 0;
                float maxLogit = float.MinValue;

                for (int i = 0; i < 512; i++)
                {
                    float val = logits[0, 63, i];
                    if (val > maxLogit)
                    {
                        maxLogit = val;
                        bestToken = i;
                    }
                }

                confidence = maxLogit; // Simplification (not actual prob)
                currentTranslation = $"TOKEN_{bestToken}"; // Labels to be mapped via decoder later
            }
        }
    }
}
