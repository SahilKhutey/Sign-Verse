import os
import tarfile
import numpy as np
import io
import json
from tqdm import tqdm
import argparse

class WebDatasetConverter:
    """
    Converts directories of gesture sequences (.npy) and labels (.txt or .json)
    into sharded .tar files for high-performance distributed training.
    """
    def __init__(self, output_dir, samples_per_shard=1000):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        os.makedirs(output_dir, exist_ok=True)

    def write_shard(self, shard_id, samples):
        """Writes a single .tar shard."""
        shard_path = os.path.join(self.output_dir, f"shard-{shard_id:06d}.tar")
        with tarfile.open(shard_path, "w") as tar:
            for i, (sample_id, data, metadata) in enumerate(samples):
                # Save .npy data
                npy_buffer = io.BytesIO()
                np.save(npy_buffer, data)
                npy_buffer.seek(0)
                
                info = tarfile.TarInfo(name=f"{sample_id}.npy")
                info.size = npy_buffer.getbuffer().nbytes
                tar.addfile(info, npy_buffer)
                
                # Save .json metadata (translation, label, etc.)
                json_data = json.dumps(metadata).encode("utf-8")
                json_buffer = io.BytesIO(json_data)
                
                info = tarfile.TarInfo(name=f"{sample_id}.json")
                info.size = len(json_data)
                tar.addfile(info, json_buffer)
                
        return shard_path

    def convert_directory(self, data_dir, label_file=None):
        """
        Iterates through a directory and shards its contents.
        Assumes data_dir contains .npy files.
        """
        all_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        print(f"Found {len(all_files)} samples. Sharding into groups of {self.samples_per_shard}...")
        
        # Mock metadata if no label file provided
        samples = []
        shard_id = 0
        
        for i, fname in enumerate(tqdm(all_files)):
            file_path = os.path.join(data_dir, fname)
            sample_id = os.path.splitext(fname)[0]
            
            data = np.load(file_path)
            metadata = {"id": sample_id, "label": "UNKNOWN", "text": "Sample translation"} 
            
            samples.append((sample_id, data, metadata))
            
            if len(samples) >= self.samples_per_shard:
                self.write_shard(shard_id, samples)
                samples = []
                shard_id += 1
                
        # Write remaining samples
        if samples:
            self.write_shard(shard_id, samples)
            
        print(f"Conversion complete. Created {shard_id + 1} shards in {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebDataset Converter for SignVerse")
    parser.add_argument("--input", type=str, required=True, help="Directory containing .npy files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .tar shards")
    parser.add_argument("--shard-size", type=int, default=1000, help="Samples per shard")
    args = parser.parse_args()
    
    converter = WebDatasetConverter(args.output, args.shard_size)
    converter.convert_directory(args.input)
