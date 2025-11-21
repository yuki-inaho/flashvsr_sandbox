#!/bin/bash
# Block-Sparse Attention Installation Script for FlashVSR
# This script builds and installs Block-Sparse Attention with CUDA 11.8 support

set -e  # Exit on error

echo "========================================="
echo "Block-Sparse Attention Installation"
echo "========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$WORKSPACE_ROOT/build"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone Block-Sparse Attention if not already cloned
if [ ! -d "Block-Sparse-Attention" ]; then
    echo "Cloning Block-Sparse Attention repository..."
    git clone https://github.com/mit-han-lab/Block-Sparse-Attention
else
    echo "Block-Sparse Attention repository already exists."
fi

cd Block-Sparse-Attention

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for package installation..."
    UV_PREFIX="uv pip install"
else
    echo "uv not found, using pip..."
    UV_PREFIX="pip install"
fi

# Install build dependencies
echo "Installing build dependencies..."
$UV_PREFIX packaging
$UV_PREFIX ninja

# Limit ninja parallel jobs to reduce memory consumption
# This is important for systems with limited memory
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

echo "Building Block-Sparse Attention..."
echo "Note: This may take 10-30 minutes depending on your system."
echo "Memory usage will be significant during compilation."

# Build and install
python setup.py install

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import block_sparse_attn; print('Block-Sparse Attention successfully installed!')" && \
    echo "✓ Installation successful!" || \
    echo "✗ Installation failed. Please check the error messages above."

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
