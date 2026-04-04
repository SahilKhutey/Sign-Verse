# SignVerse AI Architecture Design

This document describes the foundational architecture of the SignVerse AI system, designed for research-grade scalability (10B+ parameters) and real-time inference (30+ FPS).

## 1. Multi-modal Foundation Model (SFM)
The core architecture is a **Temporal Multi-modal Transformer** that maps 848-dimensional motion intelligence vectors into both sign-to-text (translation) and text-to-motion (production) spaces.

- **Encoder**: 12-layer Transformer with 1024 hidden units and 16 attention heads.
- **Input**: 848-dim vectors (Position + Velocity).
- **Latency**: Optimized for <30ms forward pass on NVIDIA H100.

## 2. Vision Pipeline (High Fidelity)
Using **MediaPipe Holistic (543 landmarks)** with high-performance C++ wrappers and Python bindings for CUDA-compatible tracking.

- **Feature Engineering**: Pairwise hand distances (210/hand), body orientation (3), and facial expression markers.
- **Throughput**: Sustains 30+ FPS via decoupled frame capture and processing threads.

## 3. Real-time Inference Server
High-performance **FastAPI** server with asynchronous WebSocket handlers and production-grade security.

- **Security**: JWT token mandatory for production streaming. AES-256 encryption for any stored biometric metadata.
- **Scalability**: Designed for Horizontal Pod Autoscaling (HPA) in Kubernetes with GPU-sharing (NVIDIA MPS).

## 4. Dataset Auto-Builder
An automated pipeline to extract and align sign language data from public repositories using **yt-dlp** and **Whisper**.

- **Workflow**: YouTube/Web -> Download -> Speech-to-Text alignment -> Landmark extraction -> WebDataset sharding.

## 5. Unity XR Bridge
Low-latency C# integration for Unity, enabling real-time avatar animation in virtual and augmented reality.

- **Communication**: WebSocket protocol with JSON performance-optimized serialization.
- **Retargeting**: Humanoid rig mapper that translates 848-dim vectors into standard Unity Animator parameters.

## Directory Mapping to System Requirements

| Requirement | Code Path |
| :--- | :--- |
| **Real-time Recognition (30+ FPS)** | `vision_system/holistic_tracker.py` |
| **Multi-modal Transformer** | `ai_models/foundation/sign_transformer.py` |
| **Secure API (JWT/Rate Limit)** | `api_server/core/security.py`, `api_server/server.py` |
| **Dataset Auto-Builder** | `dataset_builder/auto_builder.py` |
| **Unity Integration** | `ar-vr-app/unity-project/Assets/Scripts/SignVerse/` |
| **GPU Deployment** | `deployment/docker/Dockerfile.gpu` |
| **DDP Training** | `training/dist_train.py` |
