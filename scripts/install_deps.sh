#!/bin/bash

# Dependency installation script
set -e

echo "Installing system dependencies..."

# Ubuntu/Debian
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y \
        python3.10 \
        python3.10-venv \
        python3-pip \
        nodejs \
        npm \
        redis-server \
        ffmpeg \
        libgl1 \
        libglib2.0-0

# macOS
elif command -v brew &> /dev/null; then
    brew update
    brew install \
        python@3.10 \
        node \
        redis \
        ffmpeg

else
    echo "Unsupported operating system. Please install dependencies manually."
    exit 1
fi

echo "Dependencies installed successfully!"
