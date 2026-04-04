from ai_engine.modules.speech_to_text import SpeechToText
from ai_engine.modules.text_to_sign import TextToSignConverter


class InferencePipeline:

    def __init__(self):

        self.speech = SpeechToText()
        self.text_to_sign = TextToSignConverter()

    def speech_to_sign(self, audio_path):

        text = self.speech.transcribe(audio_path)

        tokens = self.text_to_sign.convert(text)

        return {
            "text": text,
            "sign_tokens": tokens
        }
