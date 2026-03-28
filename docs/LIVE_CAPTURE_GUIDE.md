# Live Capture Guide (OpenCV)

This guide explains how to run the local webcam capture pipeline and improve tracking quality.

## Run Live Capture

```
python scripts/run_live_opencv_translate.py --camera 0
```

Press `q` to exit.

## Helpful Options

```
python scripts/run_live_opencv_translate.py --list-cameras
python scripts/run_live_opencv_translate.py --camera 1 --width 1280 --height 720 --fps 20 --flip
python scripts/run_live_opencv_translate.py --camera 0 --min-confidence 0.6 --no-guides
```

Flags:
- `--list-cameras`: Probe and print available camera indices.
- `--width` / `--height`: Set capture resolution.
- `--fps`: Cap capture loop FPS.
- `--flip`: Mirror the camera for easier signing.
- `--min-confidence`: Hide labels below a confidence threshold.
- `--no-guides`: Disable framing guides overlay.
- `--no-fps`: Hide the FPS overlay.

## Web Live Stream

The web UI uses WebSocket streaming:
- Start stream in the Live Translator UI.
- Frames are captured every ~100ms and sent as JPEG (binary WebSocket).
- The server enforces a 1MB payload limit and FPS throttling.
- If `STREAM_TOKEN_REQUIRED=true`, pass `VITE_STREAM_TOKEN` for authorization.

## Debug Overlay

Use the debug endpoint to get annotated frames:
`POST /vision/extract-debug` returns an `image_b64` overlay.

## Tracking Guidelines

1. **Lighting**
   - Use bright, even lighting.
   - Avoid strong backlight or shadows.

2. **Framing**
   - Keep both hands fully visible.
   - Position your torso and shoulders in frame for pose landmarks.

3. **Distance**
   - Stand about 1-2 meters from the camera.
   - Hands should occupy 20-40% of the frame height.

4. **Background**
   - Prefer a plain background.
   - Avoid motion behind you.

5. **Speed**
   - Start slow, then increase speed.
   - Sudden fast gestures reduce accuracy.

## Gesture Points + Tracking

The system uses the canonical **225-dim keypoint layout**:

- Body (33 × 3 = 99)
- Left Hand (21 × 3 = 63)
- Right Hand (21 × 3 = 63)

Landmarks are tracked frame-by-frame and smoothed using a temporal filter
to reduce noisy predictions.

## Output

The overlay shows the **smoothed gesture ID or label**.
If no label map is available, the ID is displayed instead.
