using UnityEngine;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Management;

namespace SignVerse
{
    /// <summary>
    /// XRHandSetup — Initializes and configures OpenXR Hand Tracking.
    /// Ensures that hand tracking is active before the AI systems start.
    /// </summary>
    public class XRHandSetup : MonoBehaviour
    {
        private XRHandSubsystem _handSubsystem;

        private async void Awake()
        {
            // Verify XR Management is active
            if (XRGeneralSettings.Instance == null || XRGeneralSettings.Instance.Manager == null)
            {
                Debug.LogWarning("[SignVerse] XR Manager not active. Waiting...");
                return;
            }

            // Await XR Loader initialization
            await System.Threading.Tasks.Task.Delay(100);

            _handSubsystem = GetHandSubsystem();
            if (_handSubsystem != null)
            {
                Debug.Log("[SignVerse] OpenXR Hand Tracking Subsystem Initialized.");
            }
            else
            {
                Debug.LogError("[SignVerse] Failed to find XR Hand Subsystem. Check OpenXR Plugin settings.");
            }
        }

        private XRHandSubsystem GetHandSubsystem()
        {
            var loader = XRGeneralSettings.Instance.Manager.activeLoader;
            if (loader != null)
            {
                return loader.GetLoadedSubsystem<XRHandSubsystem>();
            }
            return null;
        }

        public bool IsHandTracked(Handedness handedness)
        {
            if (_handSubsystem == null) return false;
            
            var hand = (handedness == Handedness.Left) ? _handSubsystem.leftHand : _handSubsystem.rightHand;
            return hand.isTracked;
        }

        public XRHand GetHand(Handedness handedness)
        {
            if (_handSubsystem == null) return default;
            return (handedness == Handedness.Left) ? _handSubsystem.leftHand : _handSubsystem.rightHand;
        }
    }
}
