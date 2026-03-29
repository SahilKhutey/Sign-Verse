
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from nlp_translation.tokenizer import SignTokenizer
from nlp_translation.sign_grammar_converter import SignGrammarConverter

def test_tokenizer():
    print("--- Testing Tokenizer ---")
    tokenizer = SignTokenizer()
    tags = ["<2ASL>", "<2DGS>", "<2TSL>", "<2ISL>", "<2LSA>"]
    for tag in tags:
        if tag in tokenizer.word2idx:
            print(f"✓ Tag {tag} found in vocabulary (ID: {tokenizer.word2idx[tag]})")
        else:
            print(f"✗ Tag {tag} NOT found in vocabulary")

def test_grammar():
    print("\n--- Testing Grammar Converter ---")
    converter = SignGrammarConverter()
    test_cases = [
        ("I go to school yesterday", "ASL"),
        ("I go to school yesterday", "DGS"),
        ("I go to school yesterday", "TSL"),
    ]
    
    for text, lang in test_cases:
        result = converter.convert(text, lang_hint=lang)
        print(f"Input: '{text}' ({lang})")
        print(f"Output: {result}")
        if result and result[0] == f"<2{lang.upper()}>":
            print(f"✓ Correct tag prepended: {result[0]}")
        else:
            print(f"✗ Incorrect or missing tag")

if __name__ == "__main__":
    test_tokenizer()
    test_grammar()
