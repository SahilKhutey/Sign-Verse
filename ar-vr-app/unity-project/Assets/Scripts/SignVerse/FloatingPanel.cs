using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

namespace SignVerse
{
    /// <summary>
    /// FloatingPanel — A 3D UI component that can be "pushed" or "pinched".
    /// Integration with XR Interaction Toolkit for hand-based interaction.
    /// </summary>
    public class FloatingPanel : MonoBehaviour
    {
        [Header("Settings")]
        public float followSpeed = 10f;
        public float rotationSpeed = 5f;

        private XRSimpleInteractable _interactable;
        private IXRInteractor _grabInteractor;
        private bool _isGrabbed = false;
        private Camera _mainCamera;

        void Awake()
        {
            _mainCamera = Camera.main;
            _interactable = GetComponent<XRSimpleInteractable>();
            if (_interactable == null)
            {
                _interactable = gameObject.AddComponent<XRSimpleInteractable>();
            }

            _interactable.selectEntered.AddListener(OnSelectEntered);
            _interactable.selectExited.AddListener(OnSelectExited);
        }

        private void OnSelectEntered(SelectEnterEventArgs args)
        {
            _grabInteractor = args.interactorObject;
            _isGrabbed = true;
        }

        private void OnSelectExited(SelectExitEventArgs args)
        {
            _grabInteractor = null;
            _isGrabbed = false;
        }

        void Update()
        {
            if (_isGrabbed && _grabInteractor != null)
            {
                // Smoothly follow the hand (pinch-drag)
                transform.position = Vector3.Lerp(transform.position, _grabInteractor.transform.position, Time.deltaTime * followSpeed);
                // Rotate to face the user
                Quaternion targetRotation = Quaternion.LookRotation(transform.position - _mainCamera.transform.position);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.deltaTime * rotationSpeed);
            }
        }
    }
}
