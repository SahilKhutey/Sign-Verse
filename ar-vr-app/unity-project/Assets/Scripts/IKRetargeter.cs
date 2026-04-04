using System.Collections.Generic;
using UnityEngine;

public class IKRetargeter : MonoBehaviour
{
    [Header("Avatar Settings")]
    public Animator avatarAnimator;
    public Transform rootTransform;

    [Header("Retargeting Settings")]
    public float smoothing = 15f;
    public bool isLeftHandActive = true;
    public bool isRightHandActive = true;

    // Mapping of Joint ID to Avatar Transform
    private Dictionary<int, Transform> boneMap = new Dictionary<int, Transform>();

    void Start()
    {
        if (avatarAnimator == null) avatarAnimator = GetComponent<Animator>();
        InitializeBoneMap();
    }

    void InitializeBoneMap()
    {
        // Example mapping for a standard Humanoid Avatar
        // Indices follow MediaPipe / XR Hand standard (0-20 per hand)
        
        // Right Hand (MediaPipe indices 0-20)
        MapBone(0, HumanBodyBones.RightHand);
        MapBone(4, HumanBodyBones.RightThumbDistal);
        MapBone(8, HumanBodyBones.RightIndexDistal);
        MapBone(12, HumanBodyBones.RightMiddleDistal);
        MapBone(16, HumanBodyBones.RightRingDistal);
        MapBone(20, HumanBodyBones.RightLittleDistal);

        // Left Hand (MediaPipe indices 21-41)
        MapBone(21, HumanBodyBones.LeftHand);
        MapBone(25, HumanBodyBones.LeftThumbDistal);
        MapBone(29, HumanBodyBones.LeftIndexDistal);
        MapBone(33, HumanBodyBones.LeftMiddleDistal);
        MapBone(37, HumanBodyBones.LeftRingDistal);
        MapBone(41, HumanBodyBones.LeftLittleDistal);
    }

    void MapBone(int jointIndex, HumanBodyBones bone)
    {
        Transform t = avatarAnimator.GetBoneTransform(bone);
        if (t != null) boneMap[jointIndex] = t;
    }

    /// <summary>
    /// Call this from GestureReceiver or MediaPipe client when new coordinates arrive.
    /// joints: Dictionary<int, Vector3> where int is MediaPipe index.
    /// </summary>
    public void ApplyJointData(Dictionary<int, Vector3> joints)
    {
        foreach (var joint in joints)
        {
            if (boneMap.TryGetValue(joint.Key, out Transform boneTransform))
            {
                // Convert relative landmark (0-1) to world space or local space
                // This assumes 'rootTransform' is the anchor for coordinates
                Vector3 worldPos = rootTransform.TransformPoint(joint.Value);

                // Apply with smoothing
                boneTransform.position = Vector3.Lerp(
                    boneTransform.position, 
                    worldPos, 
                    Time.deltaTime * smoothing
                );
            }
        }
    }
}
