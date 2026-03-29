using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace SignVerse
{
    [System.Serializable]
    public class TokenizerData
    {
        public int n_clusters;
        public int feature_dim;
        public float[][] centroids;
    }

    /// <summary>
    /// GestureTokenizer — C# Vector Quantization for Sign Language.
    /// Maps 848-dim motion intelligence vectors to discrete token IDs using the KMeans codebook.
    /// </summary>
    public class GestureTokenizer
    {
        private float[,] _centroids;
        private int _numClusters;
        private int _featureDim;

        public bool IsLoaded { get; private set; }

        public void LoadVocabulary(string jsonPath)
        {
            if (!File.Exists(jsonPath))
            {
                Debug.LogError($"[SignVerse] Vocabulary file not found at: {jsonPath}");
                return;
            }

            string json = File.ReadAllText(jsonPath);
            TokenizerData data = JsonUtility.FromJson<TokenizerData>(json);

            if (data == null || data.centroids == null)
            {
                Debug.LogError("[SignVerse] Failed to parse gesture_vocabulary.json.");
                return;
            }

            _numClusters = data.n_clusters;
            _featureDim = data.feature_dim;
            _centroids = new float[_numClusters, _featureDim];

            for (int i = 0; i < _numClusters; i++)
            {
                for (int j = 0; j < _featureDim; j++)
                {
                    _centroids[i, j] = data.centroids[i][j];
                }
            }

            IsLoaded = true;
            Debug.Log($"[SignVerse] Gesture vocabulary loaded: {_numClusters} tokens.");
        }

        public int Tokenize(float[] features)
        {
            if (!IsLoaded) return 0;

            int bestCluster = 0;
            float minDistanceSq = float.MaxValue;

            // Optimization: Find nearest neighbor using squared Euclidean distance (avoids Sqrt)
            for (int i = 0; i < _numClusters; i++)
            {
                float distSq = 0;
                for (int j = 0; j < _featureDim; j++)
                {
                    float diff = features[j] - _centroids[i, j];
                    distSq += diff * diff;
                }

                if (distSq < minDistanceSq)
                {
                    minDistanceSq = distSq;
                    bestCluster = i;
                }
            }

            return bestCluster;
        }
    }
}
