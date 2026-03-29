using UnityEngine;
using UnityEngine.XR.Hands;
using System.Collections.Generic;

namespace SignVerse
{
    /// <summary>
    /// EdgeMotionIntelligence — High Fidelity Sign Feature Engineering in C#.
    /// Transforms XRHand landmarks into 848-dim motion intelligence vectors.
    /// </summary>
    public class EdgeMotionIntelligence
    {
        private float[] _lastFeatures = new float[424];
        private bool _firstFrame = true;

        /// <summary>
        /// Process current hand and pose data into the 848-dim vector.
        /// </summary>
        public float[] ProcessFrame(XRHand leftHand, XRHand rightHand, Transform head, Transform leftShoulder, Transform rightShoulder)
        {
            float[] currentFeatures = Extract424Dim(leftHand, rightHand, leftShoulder, rightShoulder);
            float[] velocity = new float[424];

            if (!_firstFrame)
            {
                for (int i = 0; i < 424; i++)
                {
                    velocity[i] = currentFeatures[i] - _lastFeatures[i];
                }
            }
            else
            {
                _firstFrame = false;
            }

            // Update last features
            System.Array.Copy(currentFeatures, _lastFeatures, 424);

            // Concatenate Position + Velocity = 848 dims
            float[] motionVector = new float[848];
            System.Array.Copy(currentFeatures, 0, motionVector, 0, 424);
            System.Array.Copy(velocity, 0, motionVector, 424, 424);

            return motionVector;
        }

        private float[] Extract424Dim(XRHand leftHand, XRHand rightHand, Transform lShoulder, Transform rShoulder)
        {
            float[] features = new float[424];
            
            // 1. Left Hand (210 dims)
            float[] lhDist = ComputeHandDistances(leftHand);
            System.Array.Copy(lhDist, 0, features, 0, 210);

            // 2. Right Hand (210 dims)
            float[] rhDist = ComputeHandDistances(rightHand);
            System.Array.Copy(rhDist, 0, features, 210, 210);

            // 3. Body Orientation (3 dims)
            Vector3 body = ComputeBodyOrientation(lShoulder, rShoulder);
            features[420] = body.x;
            features[421] = body.y;
            features[422] = body.z;

            // 4. Mouth Aperture (Dummy for now if face tracking not active, 1 dim)
            features[423] = 0.0f; // Placeholder as standardized MP Face Mesh not always in OpenXR

            return features;
        }

        private float[] ComputeHandDistances(XRHand hand)
        {
            float[] dists = new float[210];
            if (!hand.isTracked) return dists;

            // Get all 21 joints
            List<Vector3> joints = new List<Vector3>();
            for (int i = (int)XRHandJointID.Palm; i <= (int)XRHandJointID.LittleTip; i++)
            {
                if (hand.GetJoint((XRHandJointID)i).TryGetPose(out Pose pose))
                {
                    joints.Add(pose.position);
                }
                else
                {
                    joints.Add(Vector3.zero);
                }
            }

            int index = 0;
            for (int i = 0; i < joints.Count; i++)
            {
                for (int j = i + 1; j < joints.Count; j++)
                {
                    if (index < 210)
                    {
                        dists[index++] = Vector3.Distance(joints[i], joints[j]);
                    }
                }
            }
            return dists;
        }

        private Vector3 ComputeBodyOrientation(Transform lShoulder, Transform rShoulder)
        {
            if (lShoulder == null || rShoulder == null) return Vector3.zero;
            Vector3 direction = rShoulder.position - lShoulder.position;
            return direction.normalized;
        }
    }
}
