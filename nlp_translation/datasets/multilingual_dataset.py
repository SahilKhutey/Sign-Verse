import torch
from torch.utils.data import Dataset
import numpy as np

class MultilingualSignDataset(Dataset):
    """
    Dataset wrapper that injects language tags into sequences.
    Supports ASL, ISL, BSL.
    """
    def __init__(self, sign_sequences, translations, languages, tokenizer):
        self.sequences = sign_sequences
        self.translations = translations
        self.languages = languages
        self.tokenizer = tokenizer
        
        self.lang_to_tag = {
            "ASL": tokenizer.ASL_TAG,
            "DGS": tokenizer.DGS_TAG,
            "TSL": tokenizer.TSL_TAG,
            "ISL": tokenizer.ISL_TAG,
            "LSA": tokenizer.LSA_TAG
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        text = self.translations[idx]
        lang = self.languages[idx]
        
        # Prepend language tag to the text translation
        tagged_text = f"{self.lang_to_tag.get(lang.upper(), '')} {text}"
        tokens = self.tokenizer.encode(tagged_text)
        
        return {
            "sequence": seq,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "language": lang
        }
