import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import webdataset as wds
from models.gesture_ai import GestureTransformer
import os

def setup():
    """Initializes distributed process group."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    """Cleans up distributed process group."""
    dist.destroy_process_group()

def create_distributed_wds_loader(urls, batch_size=32):
    """
    Creates a distributed WebDataset loader for Pose sequences.
    Optimized for multi-node/GPU training.
    """
    dataset = (
        wds.WebDataset(urls, resampled=True)
        .shuffle(1000)
        .decode("numpy")
        .to_tuple("npy", "json")
        .map(lambda x: (torch.tensor(x[0], dtype=torch.float32), x[1]))
    )
    
    loader = wds.WebLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=4,
        pin_memory=True
    )
    
    # Optional: distributed sampler for exact epoch control 
    # instead of resampling
    # sampler = DistributedSampler(dataset)
    return loader

def train_one_epoch(model, loader, optimizer, loss_fn, rank):
    """Runs a single training epoch with DDP."""
    model.train()
    total_loss = 0.0
    for batch_idx, (x, metadata) in enumerate(loader):
        # Flatten target label from metadata for classification blueprint
        # In production, this would be a real label dictionary mapping
        y = torch.randint(0, 100, (x.size(0),)).cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if rank == 0 and batch_idx % 10 == 0:
            print(f"Batch {batch_idx} Loss: {loss.item():.4f}")
            
    return total_loss / (batch_idx + 1)

def main():
    setup()
    rank = int(os.environ["LOCAL_RANK"])
    
    # Model Setup
    model = GestureTransformer(
        input_dim=225, 
        model_dim=256, 
        num_heads=8, 
        num_layers=6, 
        num_classes=100
    ).cuda()
    model = DDP(model, device_ids=[rank])
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Scaling Ready: Load from WebDataset shards
    shard_urls = "datasets/shards/shard-{000000..000100}.tar"
    loader = create_distributed_wds_loader(shard_urls, batch_size=32)
    
    for epoch in range(50):
        # With wds.resampled=True, we don't need sampler.set_epoch
        avg_loss = train_one_epoch(model, loader, optimizer, loss_fn, rank)
        
        if rank == 0:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
            # Checkpoint
            if epoch % 5 == 0:
                torch.save(model.module.state_dict(), f"checkpoints/gesture_transformer_e{epoch}.pt")

    cleanup()

if __name__ == "__main__":
    main()
