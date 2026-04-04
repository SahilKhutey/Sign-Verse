import torch
import numpy as np
import os
import json
import glob
from ai_models.tokenization.motion_tokenizer import MotionVQVAE

def tokenize_shards(input_dir, model_path, output_dir):
    """
    SignVerse Sharding Script — 10B Dataset Pipeline.
    Converts 848-dim vectors into discrete gesture token sequences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionVQVAE(num_embeddings=2048).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each shard or subdirectory
    for shard_path in glob.glob(os.path.join(input_dir, "*.npy")):
        print(f"Tokenizing shard: {shard_path}")
        data = np.load(shard_path)
        x = torch.from_numpy(data).float().to(device)
        
        with torch.no_grad():
            tokens = model.encode(x)
            
        # Saved as discrete token sequences (B, T)
        token_data = tokens.view(x.shape[0], -1).cpu().numpy()
        output_path = os.path.join(output_dir, os.path.basename(shard_path).replace(".npy", "_tokens.npy"))
        np.save(output_path, token_data)
        print(f"Tokens saved to {output_path}")

if __name__ == "__main__":
    # Example usage (Mock)
    pass
