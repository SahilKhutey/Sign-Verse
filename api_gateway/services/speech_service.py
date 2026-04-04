"""
Speech Service — Backend logic for speech processing.
"""

class SpeechServiceHandler:

    @staticmethod
    def transcribe(audio_path):
        from ai_engine.modules.speech_to_text import SpeechToText
        stt = SpeechToText()
        text = stt.transcribe(audio_path)
        return {"text": text}

    @staticmethod
    def speech_to_sign(audio_path):
        from ai_engine.inference_pipeline import InferencePipeline
        pipeline = InferencePipeline()
        return pipeline.speech_to_sign(audio_path)
