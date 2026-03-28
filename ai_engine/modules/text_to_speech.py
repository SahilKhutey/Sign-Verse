import base64
import os
import tempfile


class TextToSpeech:
    """
    Text-to-speech module.

    Uses pyttsx3 for offline TTS. If unavailable, raises ModuleNotFoundError.
    """

    def __init__(self, voice_name: str | None = None, rate: int | None = None):
        try:
            import pyttsx3  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'pyttsx3'. Install requirements.txt to use TextToSpeech."
            ) from e
        self._engine = pyttsx3.init()
        if rate is not None:
            self._engine.setProperty("rate", int(rate))
        if voice_name:
            for v in self._engine.getProperty("voices"):
                if voice_name.lower() in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break

    def synthesize(self, text: str, output_path: str | None = None):
        if not text:
            return {"audio_path": None, "audio_bytes": None, "audio_b64": None}

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        self._engine.save_to_file(text, output_path)
        self._engine.runAndWait()

        audio_bytes = None
        try:
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
        except Exception:
            audio_bytes = None

        audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        return {"audio_path": output_path, "audio_bytes": audio_bytes, "audio_b64": audio_b64}
