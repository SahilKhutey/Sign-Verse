"""
Translation Service — Backend logic for text translation.
"""

class TranslationServiceHandler:

    @staticmethod
    def text_to_sign(text):
        from nlp_translation.text_to_gloss import text_to_gloss
        tokens = text_to_gloss(text)
        return {"text": text, "sign_tokens": tokens}

    @staticmethod
    def sign_to_text(gloss):
        from nlp_translation.gloss_to_text import gloss_to_text
        return {"text": gloss_to_text(gloss)}

    @staticmethod
    def batch(texts):
        from nlp_translation.text_to_gloss import text_to_gloss
        return [{"text": t, "sign_tokens": text_to_gloss(t)} for t in texts]
