#!/bin/bash

# SignVerse System Setup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up SignVerse System...${NC}"

# Check if Python 3.10+ is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3.10+ is required but not installed.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo -e "${RED}Python 3.10+ is required. Found Python $PYTHON_VERSION${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install Node.js dependencies for frontend
echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
cd frontend
npm install
cd ..

# Create necessary directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data/raw/uploads
mkdir -p data/raw/youtube_videos
mkdir -p data/processed/frames
mkdir -p data/processed/poses
mkdir -p data/processed/annotations
mkdir -p data/labeled/actions
mkdir -p data/labeled/joints
mkdir -p data/datasets
mkdir -p data/metadata
mkdir -p models/checkpoints
mkdir -p models/experiments
mkdir -p logs

# Copy environment template
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please update .env file with your configuration${NC}"
fi

# Set up git hooks
echo -e "${YELLOW}Setting up git hooks...${NC}"
cp scripts/git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Initialize git LFS (if available)
if command -v git-lfs &> /dev/null; then
    echo -e "${YELLOW}Initializing Git LFS for large files...${NC}"
    git lfs install
    git lfs track "data/**/*.mp4"
    git lfs track "data/**/*.avi"
    git lfs track "data/**/*.mov"
    git lfs track "models/checkpoints/*.pt"
    git lfs track "models/checkpoints/*.pth"
fi

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update .env file with your configuration"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python scripts/download_data.py --sample"
echo "4. Run: python -m pytest tests/unit/ -v"
