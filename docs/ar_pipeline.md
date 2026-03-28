# AR/VR Pipeline

Goal: stream recognized sign information and generated motion to an AR/VR client (Unity) in real time.

## Current State in Repo

- Client app folder: `ar-vr-app/`
  - Contains Unity project structure and mobile/web frontends (not analyzed deeply in this pass).
- Server streaming endpoint:
  - `api_server/server.py` exposes `WS /ws/stream`.
  - `api_server/realtime_inference.py` implements `process_stream_frame()` which currently:
    - classifies a gesture from keypoints
    - converts any provided text to sign tokens (rule-based)
    - returns `gesture_label` (if label_map.json is available) and uses it as a fallback `sign_tokens` when text is empty
    - returns a JSON payload per frame
- Avatar animation:
  - `avatar_animation/` exists, but many controllers/engine files are currently 0-byte stubs and need implementation.

## Target Runtime Flow (Recommended)

1. AR/VR client captures camera frames (or receives keypoints from a device-side pose estimator).
2. Client sends keypoints to server via WebSocket:
   - payload includes `frame_id` and `keypoints`
3. Server outputs:
   - recognized gesture tokens/IDs
   - (optional) translated text
   - (optional) generated motion vectors for the avatar
4. Unity client:
   - retargets motion vectors to a humanoid rig
   - blends facial expressions and hand articulation

## Remaining Work

- Define a stable websocket schema:
  - keypoint schema name, dimensionality, fps, coordinate conventions
- Implement avatar retargeting and animation blending:
  - motion vectors -> rig joints
  - hand/finger articulation
  - facial grammar support (questions, emphasis)
- Measure end-to-end latency and add buffering/smoothing controls.
