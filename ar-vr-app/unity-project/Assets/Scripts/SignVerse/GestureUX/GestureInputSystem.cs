using UnityEngine;
using UnityEngine.XR.Hands;
using System;
using System.Collections.Generic;

namespace SignVerse.GestureUX
{
    /// <summary>
    /// GestureCommand — Enumeration of all navigable UI commands.
    /// Maps gesture patterns to semantic actions. No buttons involved.
    /// </summary>
    public enum GestureCommand
    {
        None,
        Select,          // Pinch (index + thumb)
        Back,            // Swipe left (palm out, move left)
        ScrollUp,        // Open palm swipe up
        ScrollDown,      // Open palm swipe down
        OpenMenu,        // Both hands form an "O" shape
        CloseMenu,       // Both hands flat and move together
        ConfirmAction,   // Thumbs up (right hand)
        DismissPanel,    // Flick wrist outward
    }

    /// <summary>
    /// GestureInputSystem — Detects and dispatches gesture-based UI commands.
    ///
    /// Design principles:
    ///   - Single-hand gestures for primary navigation (less cognitive load)
    ///   - Dual-hand gestures only for destructive or global actions
    ///   - Dwell + velocity thresholds prevent accidental triggers
    ///   - 300ms cooldown between commands (safety gate)
    /// </summary>
    public class GestureInputSystem : MonoBehaviour
    {
        [Header("XR Hands")]
        public XRHandSubsystem handSubsystem;

        [Header("Safety Thresholds")]
        [Tooltip("Pinch distance (meters) to register Select")]
        public float pinchDistanceThreshold   = 0.035f; // Increased for easier pinching
        [Tooltip("Min swipe velocity (m/s) to register directional gestures")]
        public float swipeVelocityThreshold   = 0.35f;  // Lowered for easier swipes
        [Tooltip("Gesture must be held this long (sec) to prevent accidents")]
        public float dwellTime                = 0.15f;  // Slightly longer dwell to avoid accidental triggers
        [Tooltip("Cooldown between any two commands (sec)")]
        public float commandCooldown          = 0.30f;

        // Events
        public event Action<GestureCommand> OnGestureCommand;

        // ── Private state ──────────────────────────────────────────────────────
        private float   _lastCommandTime   = -999f;
        private float   _pinchHeldSince    = -1f;
        private float   _thumbsUpHeldSince = -1f;

        private Vector3 _prevRightPalmPos;
        private Vector3 _prevLeftPalmPos;
        private float   _prevFrameTime;

        // ── Lifecycle ──────────────────────────────────────────────────────────
        void Start()
        {
            _prevFrameTime = Time.time;
        }

        void Update()
        {
            float dt = Time.time - _prevFrameTime;
            _prevFrameTime = Time.time;

            if (handSubsystem == null || !handSubsystem.running) return;

            XRHand right = handSubsystem.rightHand;
            XRHand left  = handSubsystem.leftHand;

            if (!right.isTracked && !left.isTracked) return;

            // ── Detect gestures in priority order ──────────────────────────────
            DetectPinchSelect(right, left);
            DetectSwipes(right, left, dt);
            DetectThumbsUp(right);
            DetectWristFlick(right, dt);
            DetectBimanualMenuGestures(right, left);

            // Cache palm positions for velocity calculations
            if (right.isTracked && TryGetJointPos(right, XRHandJointID.Palm, out var rp)) _prevRightPalmPos = rp;
            if (left.isTracked  && TryGetJointPos(left,  XRHandJointID.Palm, out var lp)) _prevLeftPalmPos  = lp;
        }

        // ── Gesture Detectors ──────────────────────────────────────────────────

        private void DetectPinchSelect(XRHand right, XRHand left)
        {
            // Use dominant (right) hand first, fallback to left
            XRHand hand = right.isTracked ? right : (left.isTracked ? left : default);
            if (!hand.isTracked) return;

            if (!TryGetJointPos(hand, XRHandJointID.IndexTip, out var indexTip)) return;
            if (!TryGetJointPos(hand, XRHandJointID.ThumbTip, out var thumbTip)) return;

            float dist = Vector3.Distance(indexTip, thumbTip);
            if (dist < pinchDistanceThreshold)
            {
                if (_pinchHeldSince < 0) _pinchHeldSince = Time.time;
                if (Time.time - _pinchHeldSince >= dwellTime)
                {
                    Dispatch(GestureCommand.Select);
                    _pinchHeldSince = -1f;
                }
            }
            else
            {
                _pinchHeldSince = -1f;
            }
        }

        private void DetectSwipes(XRHand right, XRHand left, float dt)
        {
            if (!right.isTracked || dt <= 0) return;
            if (!TryGetJointPos(right, XRHandJointID.Palm, out var palmPos)) return;

            Vector3 velocity = (palmPos - _prevRightPalmPos) / dt;

            if (velocity.magnitude < swipeVelocityThreshold) return;

            // Vertical dominant → scroll
            if (Mathf.Abs(velocity.y) > Mathf.Abs(velocity.x) * 1.5f)
            {
                Dispatch(velocity.y > 0 ? GestureCommand.ScrollUp : GestureCommand.ScrollDown);
            }
            // Horizontal dominant → back
            else if (velocity.x < -swipeVelocityThreshold * 1.2f)
            {
                Dispatch(GestureCommand.Back);
            }
        }

        private void DetectThumbsUp(XRHand right)
        {
            if (!right.isTracked) return;
            if (!TryGetJointPos(right, XRHandJointID.ThumbTip,    out var thumbTip))  return;
            if (!TryGetJointPos(right, XRHandJointID.Palm,         out var palm))      return;
            if (!TryGetJointPos(right, XRHandJointID.IndexTip,     out var indexTip))  return;
            if (!TryGetJointPos(right, XRHandJointID.MiddleTip,    out var middleTip)) return;

            // Thumb must be above palm significantly; other fingers curled
            bool thumbUp     = thumbTip.y > palm.y + 0.04f;
            bool fingersCurled = indexTip.y < palm.y + 0.01f && middleTip.y < palm.y + 0.01f;

            if (thumbUp && fingersCurled)
            {
                if (_thumbsUpHeldSince < 0) _thumbsUpHeldSince = Time.time;
                if (Time.time - _thumbsUpHeldSince >= dwellTime)
                {
                    Dispatch(GestureCommand.ConfirmAction);
                    _thumbsUpHeldSince = -1f;
                }
            }
            else _thumbsUpHeldSince = -1f;
        }

        private void DetectWristFlick(XRHand right, float dt)
        {
            if (!right.isTracked || dt <= 0) return;
            if (!TryGetJointPos(right, XRHandJointID.Wrist, out var wrist)) return;

            Vector3 wristVelocity = (wrist - _prevRightPalmPos) / dt;
            // Fast outward flick (positive local X)
            if (wristVelocity.x > swipeVelocityThreshold * 2.5f)
            {
                Dispatch(GestureCommand.DismissPanel);
            }
        }

        private void DetectBimanualMenuGestures(XRHand right, XRHand left)
        {
            if (!right.isTracked || !left.isTracked) return;

            if (!TryGetJointPos(right, XRHandJointID.Palm, out var rPalm)) return;
            if (!TryGetJointPos(left,  XRHandJointID.Palm, out var lPalm)) return;

            float handSeparation = Vector3.Distance(rPalm, lPalm);

            // Hands close together and moving inward → CloseMenu
            Vector3 approaching = rPalm - lPalm;
            if (handSeparation < 0.15f && approaching.magnitude < 0.05f)
            {
                Dispatch(GestureCommand.CloseMenu);
            }
            // Hands spread and separation > 0.4m → OpenMenu
            else if (handSeparation > 0.40f)
            {
                Dispatch(GestureCommand.OpenMenu);
            }
        }

        // ── Dispatch ───────────────────────────────────────────────────────────
        private void Dispatch(GestureCommand cmd)
        {
            if (cmd == GestureCommand.None) return;
            if (Time.time - _lastCommandTime < commandCooldown) return;

            _lastCommandTime = Time.time;
            OnGestureCommand?.Invoke(cmd);
            Debug.Log($"[GestureUX] Command: {cmd}");
        }

        // ── Helpers ────────────────────────────────────────────────────────────
        private bool TryGetJointPos(XRHand hand, XRHandJointID joint, out Vector3 pos)
        {
            pos = Vector3.zero;
            if (hand.GetJoint(joint).TryGetPose(out Pose pose))
            {
                pos = pose.position;
                return true;
            }
            return false;
        }
    }
}
