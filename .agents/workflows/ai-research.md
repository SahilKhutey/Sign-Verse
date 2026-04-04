---
description: AI Research SOP — Training and Scaling SignGPT (10B+)
---
# AI Research Workflow

Processes for model innovation, data sharding, and distributed training of the SignVerse foundation.

## Prerequisites
- **PyTorch** and **Transformers** knowledge
- **DeepSpeed** configuration expertise
- Access to **H100/A100 GPU Cluster**

## Workflow Steps

### 1. Data Pipeline
// turbo
1.  **Auto-Builder**: Run `dataset_builder/auto_builder.py` for new sign sources.
2.  **Tokenization**: Run `training/data_pipeline/tokenize_video.py` to convert 848-dim vectors to discrete VQ-VAE tokens.
3.  **Sharding**: Use `HugeDatasetLoader` to verify WebDataset shard integrity.

### 2. Model Training (DDP/DeepSpeed)
// turbo
1.  **Config**: Update `training/configs/training_config.yaml`.
2.  **Launch**: `deepspeed --num_gpus 8 training/train_huge_model.py`.
3.  **Monitor**: Use **WandB** or **TensorBoard** via `models/checkpoints/logs`.

### 3. Evaluation
1.  **Metrics**: Track **BLEU**, **WER (Word Error Rate)**, and **Motion Reconstruction MSE**.
2.  **Latency**: Run `testing/performance_tests/realtime_latency_test.py`.

## Universal Interface (848-dim)
All models must accept or output the **848-dimensional "Motion Intelligence" vector**. If a new dimension is required, update `common/keypoint_schema.py` and notify the Backend/XR teams.
