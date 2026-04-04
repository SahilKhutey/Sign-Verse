# SignVerse AI — User & Developer Guide

Welcome to the **SignVerse AI** platform. This guide provides instructions for setting up, running, and extending the SignVerse ecosystem.

## 🚀 Architecture Overview
SignVerse follows a microservices architecture:
- **AI Engine**: Core translation logic (Sign-to-Text, Text-to-Sign, STT, TTS).
- **Backend (FastAPI)**: History persistence, User Auth, and API Gateway.
- **Frontend (React)**: High-performance dashboard for users.
- **XR Bridge (Unity)**: real-time avatar driving and generative motion.

## 🛠️ Setup Instructions

### 1. AI Engine & Training
```bash
# Install dependencies
pip install -r requirements.txt

# Launch automated training (Stages 1-10)
python training/launch_all.py
```

### 2. Backend & Database
```bash
cd backend
pip install -r requirements.txt
# Setup migrations
python -m alembic upgrade head
# Start server
python app.py
```

### 3. Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```

## 🎥 Translation Flow
1. **Sign-to-Speech**: Camera data -> Sign feature extractor -> GPT-base Translator -> TTS Engine.
2. **Speech-to-Sign**: Audio bytes -> STT -> Tokenizer -> Gesture Diffusion -> Unity Avatar.

## 🧪 Testing & Verification
- **Integration Tests**: `python tests/integration_tests.py`
- **Real-time API Test**: `python scripts/test_real_time_api.py`
- **Backend Test**: `python scripts/test_backend_integration.py`

## 📦 Production Deployment
Use the automated deployment script for Kubernetes:
```bash
./deploy_signverse.sh
```

---
© 2024 SignVerse AI Team. Empowering communication through AI.
