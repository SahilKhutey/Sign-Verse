---
description: XR Interaction SOP — Gesture-Native UX and Avatar Feedback in Unity
---
# XR Interaction Workflow

Guidelines for building immersive sign language interfaces using Unity and OpenXR.

## Prerequisites
- **Unity 2022.3+ (LTS)**
- **XR Interaction Toolkit (XRI)**
- **Meta Quest 2/3/Pro** or similar OpenXR headset

## Workflow Steps

### 1. Project Implementation
1.  **Hand Tracking**: Ensure `XRHandSetup.cs` is active in the scene.
2.  **UI Panels**: Use `FloatingPanel.cs` for 3D interactions (Poke/Pinch).
3.  **Avatar**: Attach the `SignRigMapper` to any Humanoid-compatible rig.

### 2. Interaction Design
1.  **Pinch**: Standard "select" action for floating UI.
2.  **Palm-Up**: Trigger for the `HandMenuController`.
3.  **Haptic/Audio**: Provide feedback for successful sign recognition.

### 3. Backend Integration
1.  **WebSocket**: Point `AvatarStreamer.cs` and `GestureReceiver.cs` to the API server.
2.  **Tokens**: Ensure the `authToken` is correctly set for secure streaming.

## Universal Interface (848-dim)
Unity must correctly receive and map the **848-dimensional "Motion Intelligence" vector**. Use `SignRigMapper.cs` as the translation layer between vectors and bone rotations.
