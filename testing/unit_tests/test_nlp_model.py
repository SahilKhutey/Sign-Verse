"""
Unit Tests — NLP Model
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestTextToSign(unittest.TestCase):

    def test_basic_conversion(self):
        from ai_engine.modules.text_to_sign import TextToSignConverter
        converter = TextToSignConverter()
        result = converter.convert("I am going to school")
        self.assertIn("I", result)
        self.assertIn("GOING", result)
        self.assertIn("SCHOOL", result)
        self.assertNotIn("AM", result)

    def test_stop_word_removal(self):
        from ai_engine.modules.text_to_sign import TextToSignConverter
        converter = TextToSignConverter()
        result = converter.convert("the cat is on a mat")
        self.assertNotIn("THE", result)
        self.assertNotIn("IS", result)
        self.assertNotIn("A", result)


class TestSignGrammarConverter(unittest.TestCase):

    def test_time_reordering(self):
        from nlp_translation.sign_grammar_converter import SignGrammarConverter
        converter = SignGrammarConverter()
        result = converter.convert("I am going to school tomorrow")
        # Time marker should come first
        self.assertEqual(result[0], "TOMORROW")

    def test_stop_word_removal(self):
        from nlp_translation.sign_grammar_converter import SignGrammarConverter
        converter = SignGrammarConverter()
        result = converter.convert("the boy is happy")
        self.assertNotIn("THE", result)
        self.assertNotIn("IS", result)


class TestTokenizer(unittest.TestCase):

    def test_encode_decode(self):
        from nlp_translation.tokenizer import SignTokenizer
        tok = SignTokenizer()
        tok.build_vocab(["HELLO WORLD", "THANK YOU"])
        encoded = tok.encode("HELLO WORLD")
        decoded = tok.decode(encoded)
        self.assertEqual(decoded, "HELLO WORLD")

    def test_special_tokens(self):
        from nlp_translation.tokenizer import SignTokenizer
        tok = SignTokenizer()
        self.assertEqual(tok.word2idx["<PAD>"], 0)
        self.assertEqual(tok.word2idx["<SOS>"], 1)
        self.assertEqual(tok.word2idx["<EOS>"], 2)


class TestSignToText(unittest.TestCase):

    def test_buffer_accumulation(self):
        from ai_engine.modules.sign_to_text import SignToText
        stt = SignToText()
        for i in range(5):
            result = stt.update("HELLO")
        self.assertIsNone(result)
        result = stt.update("WORLD")
        self.assertIsNotNone(result)


class TestEvaluation(unittest.TestCase):

    def test_accuracy(self):
        from nlp_translation.utils.evaluation import accuracy
        result = accuracy(["A", "B", "C"], ["A", "B", "C"])
        self.assertEqual(result, 1.0)

    def test_word_error_rate(self):
        from nlp_translation.utils.evaluation import word_error_rate
        result = word_error_rate(["A", "B", "C"], ["A", "B", "C"])
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
