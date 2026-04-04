class SpeechToText:

    def __init__(self, model_size="base"):
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'faster-whisper'. Install requirements.txt to use SpeechToText."
            ) from e
        self.model = WhisperModel(model_size)

    def transcribe(self, audio_path):

        segments, _ = self.model.transcribe(audio_path)

        text = ""
        for segment in segments:
            text += segment.text

        return text.strip()
