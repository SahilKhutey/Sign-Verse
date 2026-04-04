"""
Motion Synthesis Layer — Sign Language Production
Converts sign tokens or text into 848-dim motion sequences.
Works as a generative head for the SignTransformer.
"""

import torch
import torch.nn as nn
from ai_models.foundation.sign_transformer import SignTransformer

class MotionSynthesizer:
    def __init__(self, model_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SignTransformer().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def generate(self, initial_frame, steps=30):
        """
        Auto-regressive generation of motion vectors.
        Input: (1, 1, 848) initial frame
        Output: (1, steps, 848) generated sequence
        """
        self.model.eval()
        with torch.no_grad():
            current_frame = torch.tensor(initial_frame, dtype=torch.float32).to(self.device)
            if current_frame.ndim == 2:
                current_frame = current_frame.unsqueeze(0) # (1, 1, 848)
                
            sequence = [current_frame]
            
            for _ in range(steps - 1):
                input_seq = torch.cat(sequence, dim=1)
                _, next_frame = self.model(input_seq)
                
                # Take only the last prediction
                last_pred = next_frame[:, -1:, :]
                sequence.append(last_pred)
                
            full_seq = torch.cat(sequence, dim=1)
            return full_seq.cpu().numpy()

if __name__ == "__main__":
    # Test generation
    ms = MotionSynthesizer(device="cpu")
    dummy_start = torch.randn(1, 1, 848)
    motion = ms.generate(dummy_start, steps=10)
    print(f"Generated motion shape: {motion.shape}")
