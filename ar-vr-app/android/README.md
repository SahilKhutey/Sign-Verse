# SignVerse Launcher (Android)

This is the Android launcher app for SignVerse. It provides a complete mobile UI
and connects to the SignVerse FastAPI server for inference.

## What This App Can Do (Today)

- **Text -> Sign tokens** via `POST /translate/text-to-sign`
- **Image -> Gesture ID** (pick a photo) via:
  - `POST /analyze/frame` (image upload)
- **Live camera demo** via repeated `POST /analyze/frame` (requires camera permission)
- **Server health** via `GET /health`

## Requirements

- Android Studio (recent)
- JDK 17
- (For real inference) Run the backend server:
  - `api_server/server.py` (default: `http://<host>:8000`)

Example (from `signverse-ai/`):

```powershell
Set-Location C:\Users\User\Documents\SignVerse\signverse-ai
.\venv\Scripts\python.exe api_server\server.py
```

## Run In Android Studio

1. Open this folder as a project:
   - `signverse-ai/ar-vr-app/android/`
2. Click Run.
3. In the app, open **Settings** and set the server base URL.

Note:
- This repo includes `gradle/wrapper/gradle-wrapper.properties` (so Android Studio knows which Gradle to use),
  but does not include the `gradlew` scripts/jar. Android Studio will still sync/build normally.

Tips:
- Android emulator to your host machine:
  - Use `http://10.0.2.2:8000`
- Physical device on the same Wi-Fi:
  - Use `http://<your-pc-lan-ip>:8000`

## Build APK (Debug)

In Android Studio:
- **Build** -> **Build Bundle(s) / APK(s)** -> **Build APK(s)**

The debug APK will be under:
- `app/build/outputs/apk/debug/`
