"""
SignVerse Distributed Training — Multi-GPU DDP Runner
Optimized for 10B+ parameter scaling using NVIDIA NCCL.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from training.trainer import SignTrainer
from ai_models.foundation.sign_transformer import SignTransformer
from training.data_pipeline.preprocessor import SignPreprocessor

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_dist(rank, world_size, epochs=10):
    setup(rank, world_size)
    
    # Model
    model = SignTransformer().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Data (Distributed)
    # Using local mock patterns or WebDataset tar shards
    loader = SignPreprocessor("datasets/shards/train-*.tar", batch_size=32).get_dataloader()
    
    # Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Trainer
    trainer = SignTrainer(model.module, optimizer, criterion, device=rank)
    
    for epoch in range(epochs):
        loss = trainer.train_epoch(loader, epoch)
        if rank == 0:
            trainer.save_checkpoint(epoch, loss)
            
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No GPU found for distributed training.")
    else:
        print(f"Starting distributed training on {world_size} GPUs...")
        mp.spawn(train_dist, args=(world_size,), nprocs=world_size, join=True)
