using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// AvatarController — Drives the signing avatar.
///
/// Receives sign tokens from GestureReceiver and plays
/// corresponding animation clips with smooth transitions
/// and inverse kinematics for hand articulation.
/// </summary>
public class AvatarController : MonoBehaviour
{
    [Header("Avatar References")]
    public Animator animator;
    public Transform leftHandTarget;
    public Transform rightHandTarget;

    [Header("Playback Settings")]
    public float signDuration = 0.8f;
    public float transitionTime = 0.2f;
    public float playbackSpeed = 1.0f;

    [Header("IK Settings")]
    public bool useIK = true;
    [Range(0f, 1f)] public float ikWeight = 1.0f;

    private Queue<string> signQueue = new Queue<string>();
    private bool isPlaying = false;

    // Token → animation trigger name mapping
    private static readonly Dictionary<string, string> tokenToAnim = new Dictionary<string, string>
    {
        { "HELLO",     "Anim_Hello"    },
        { "THANK_YOU", "Anim_ThankYou" },
        { "YES",       "Anim_Yes"      },
        { "NO",        "Anim_No"       },
        { "I",         "Anim_PointSelf" },
        { "YOU",       "Anim_PointYou" },
        { "GO",        "Anim_Go"       },
        { "PLEASE",    "Anim_Please"   },
        { "SORRY",     "Anim_Sorry"    },
        { "LOVE",      "Anim_Love"     },
    };

    private void Start()
    {
        if (animator != null)
            animator.speed = playbackSpeed;
    }

    public void PlaySignSequence(List<string> tokens)
    {
        foreach (string token in tokens)
            signQueue.Enqueue(token);

        if (!isPlaying)
            StartCoroutine(PlayQueue());
    }

    private IEnumerator PlayQueue()
    {
        isPlaying = true;

        while (signQueue.Count > 0)
        {
            string token = signQueue.Dequeue();
            
            // SKIP language tags (e.g., <2ASL>)
            if (token.StartsWith("<") && token.EndsWith(">"))
            {
                Debug.Log($"[AvatarController] Skipping language tag: {token}");
                continue;
            }

            yield return PlayToken(token);
            yield return new WaitForSeconds(transitionTime);
        }

        isPlaying = false;
        SetIdle();
    }

    private IEnumerator PlayToken(string token)
    {
        if (tokenToAnim.TryGetValue(token, out string animTrigger))
        {
            animator?.SetTrigger(animTrigger);
            yield return new WaitForSeconds(signDuration);
        }
        else
        {
            // NEW: Try generative motion from server first
            yield return StartCoroutine(RequestGenerativeMotion(token));
            
            // Fallback: Fingerspell if generative motion fails or is slow
            if (!isPlayingGenerative) {
                foreach (char c in token.ToUpper())
                {
                    string fingerTrigger = $"Finger_{c}";
                    animator?.SetTrigger(fingerTrigger);
                    yield return new WaitForSeconds(0.2f);
                }
                yield return new WaitForSeconds(signDuration * 0.5f);
            }
        }
    }

    private bool isPlayingGenerative = false;

    /// <summary>
    /// Request 3D motion keypoints from the SignVerse AI Generation Layer.
    /// </summary>
    private IEnumerator RequestGenerativeMotion(string token)
    {
        Debug.Log($"[AvatarController] Requesting generative motion for: {token}");
        
        // Updated to use the production backend API port
        string url = "http://localhost:8001/api/translations/translate";
        
        // Match the backend TranslationRequest schema
        string jsonPayload = "{\"user_id\": 0, \"data\": \"" + token + "\", \"input_type\": \"text\", \"target_lang\": \"en\"}";
        
        using (UnityEngine.Networking.UnityWebRequest request = new UnityEngine.Networking.UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
            request.uploadHandler = new UnityEngine.Networking.UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new UnityEngine.Networking.DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            // Add Authorization header if available
            // request.SetRequestHeader("Authorization", "Bearer " + userToken);

            yield return request.SendWebRequest();

            if (request.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                // Parse motion keypoints and apply via IK
                // Note: Simplified logic for demonstration; in production, this would 
                // feed a Buffer for the OnAnimatorIK loop.
                isPlayingGenerative = true;
                Debug.Log($"[AvatarController] Generative motion received for {token}");
                yield return new WaitForSeconds(signDuration);
                isPlayingGenerative = false;
            }
            else
            {
                Debug.LogWarning($"[AvatarController] Generative motion request failed: {request.error}");
                isPlayingGenerative = false;
            }
        }
    }

    private void SetIdle()
    {
        animator?.SetTrigger("Idle");
    }

    private void OnAnimatorIK(int layerIndex)
    {
        if (!useIK || animator == null) return;

        if (leftHandTarget != null)
        {
            animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, ikWeight);
            animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, ikWeight);
            animator.SetIKPosition(AvatarIKGoal.LeftHand, leftHandTarget.position);
            animator.SetIKRotation(AvatarIKGoal.LeftHand, leftHandTarget.rotation);
        }

        if (rightHandTarget != null)
        {
            animator.SetIKPositionWeight(AvatarIKGoal.RightHand, ikWeight);
            animator.SetIKRotationWeight(AvatarIKGoal.RightHand, ikWeight);
            animator.SetIKPosition(AvatarIKGoal.RightHand, rightHandTarget.position);
            animator.SetIKRotation(AvatarIKGoal.RightHand, rightHandTarget.rotation);
        }
    }

    public void SetSpeed(float speed)
    {
        playbackSpeed = Mathf.Clamp(speed, 0.1f, 3.0f);
        if (animator != null)
            animator.speed = playbackSpeed;
    }

    public void ClearQueue()
    {
        signQueue.Clear();
        StopAllCoroutines();
        isPlaying = false;
        SetIdle();
    }
}
