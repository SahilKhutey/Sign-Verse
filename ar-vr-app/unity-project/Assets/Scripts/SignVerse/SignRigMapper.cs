using UnityEngine;
using System.Collections.Generic;

namespace SignVerse
{
    /// <summary>
    /// SignRigMapper — Translates 848-dim motion intelligence vectors into 
    /// Unity Humanoid rig rotations and morph targets.
    /// </summary>
    public class SignRigMapper : MonoBehaviour
    {
        [Header("Target Avatar")]
        public Animator animator;
        public SkinnedMeshRenderer faceMesh;
        public int mouthMorphIndex = 0;

        [Header("Mapping Profile")]
        public float handScale = 1.0f;
        public float smoothing = 0.5f;

        private float _mouthApeture = 0f;
        private float[] _lastVector;

        public void MapVector(float[] vector)
        {
            if (vector == null || vector.Length < 424) return;

            // 1. Mouth (Index 423 in the 424-dim geometric vector)
            _mouthApeture = Mathf.Lerp(_mouthApeture, vector[423] * 100f, smoothing);
            if (faceMesh != null)
            {
                faceMesh.SetBlendShapeWeight(mouthMorphIndex, _mouthApeture);
            }

            // 2. Hand Distances -> Finger Curls (Indices 0..209 and 210..419)
            // This is a simplified mapping: distance between finger tips and palm
            // In a full implementation, use 3D inverse kinematics or direct rotation mapping.
            MapHand(vector, true);  // Left Hand
            MapHand(vector, false); // Right Hand

            // 3. Body Orient (Indices 420, 421, 422)
            Vector3 bodyDir = new Vector3(vector[420], vector[421], vector[422]);
            if (bodyDir.sqrMagnitude > 0.001f)
            {
                Quaternion targetRot = Quaternion.LookRotation(bodyDir, Vector3.up);
                animator.transform.rotation = Quaternion.Slerp(animator.transform.rotation, targetRot, smoothing);
            }
        }

        private void MapHand(float[] vector, bool isLeft)
        {
            int offset = isLeft ? 0 : 210;
            // 210 distances per hand.
            // Simplified: distance for thumbnails, index, middle, ring, pinky
            // We'll map a subset of distances to generic Animator Hand values.
            
            // Note: In Unity, we often use animator.SetIKPosition or custom rig handles.
            // For this architectural demo, we showcase the data bridge logic.
            float thumbCurl = vector[offset + 0] * handScale;
            float pinkyCurl = vector[offset + 209] * handScale;
            
            // Apply to Animator (using hypothetical Float parameters or direct bone rot)
            string prefix = isLeft ? "Left" : "Right";
            animator.SetFloat(prefix + "ThumbCurl", thumbCurl);
            animator.SetFloat(prefix + "PinkyCurl", pinkyCurl);
        }
    }
}
