import unittest
import torch
import sys
import os

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from models.gesture_ai import GestureTransformer, MotionDiffusion
from models.v3_foundation import MultimodalFoundationModel, GestureGPT

class TestAdvancedModelsForward(unittest.TestCase):
    
    def test_gesture_transformer_shape(self):
        batch, seq_len, input_dim = 2, 30, 225
        num_classes = 100
        model = GestureTransformer(
            input_dim=input_dim, 
            model_dim=256, 
            num_heads=8, 
            num_layers=6, 
            num_classes=num_classes
        )
        model.eval()
        
        x = torch.randn(batch, seq_len, input_dim)
        with torch.no_grad():
            out = model(x)
            
        self.assertEqual(out.shape, (batch, num_classes))

    def test_motion_diffusion_shape(self):
        batch, motion_dim = 2, 225
        model = MotionDiffusion(dim=motion_dim)
        model.eval()
        
        x = torch.randn(batch, motion_dim)
        noise = torch.randn(batch, motion_dim)
        with torch.no_grad():
            pred_noise = model(x, noise)
            
        self.assertEqual(pred_noise.shape, (batch, motion_dim))

    def test_foundation_v3_shape(self):
        batch = 2
        dim = 512
        model = MultimodalFoundationModel(dim=dim, num_layers=2, num_heads=8)
        model.eval()
        
        # Dummy inputs
        video = torch.randn(batch, 10, 2048)
        pose = torch.randn(batch, 30, 225)
        text = torch.randint(0, 30000, (batch, 15))
        audio = torch.randn(batch, 50, 128)
        
        with torch.no_grad():
            fused = model(video=video, pose=pose, text=text, audio=audio)
            
        # Total tokens = 10 (video) + 30 (pose) + 15 (text) + 50 (audio) = 105
        self.assertEqual(fused.shape, (batch, 105, dim))

    def test_gesture_gpt_decoder(self):
        batch = 2
        dim = 512
        vocab = 512
        model = GestureGPT(dim=dim, vocab=vocab, num_layers=4)
        model.eval()
        
        memory = torch.randn(batch, 105, dim) # Output from foundation model
        target = torch.randint(0, vocab, (batch, 20)) # Prefix gesture tokens
        
        # Target tokens need to be embedded for the decoder (simplified in blueprint)
        # We'll mock the embedding step inside the test or adjust model if needed
        # In the blueprint, GestureGPT takes 'target' directly, but it usually 
        # needs embedding. Let's assume the blueprint implies target is already embedded 
        # or we add embedding to the model.
        
        # Let's update the model to include embedding for the 'target' indices
        target_embed = torch.randn(batch, 20, dim)
        
        with torch.no_grad():
            # Adjusting call based on current model impl (which expects embedded target)
            # Actually, the blueprint impl in v3_foundation.py used:
            # def forward(self, memory, target):
            #    out = self.decoder(target, memory)
            #    return self.head(out)
            # This requires target to be (batch, seq, dim)
            
            out = model(memory, target_embed)
            
        self.assertEqual(out.shape, (batch, 20, vocab))

if __name__ == "__main__":
    unittest.main()
