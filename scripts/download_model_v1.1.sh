#!/bin/bash
# FlashVSR v1.1 Model Download Script
# Downloads the FlashVSR-v1.1 model from HuggingFace

set -e  # Exit on error

echo "========================================="
echo "FlashVSR v1.1 Model Download"
echo "========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
FLASHVSR_DIR="$WORKSPACE_ROOT/FlashVSR"
WANVSR_DIR="$FLASHVSR_DIR/examples/WanVSR"
MODEL_DIR="$WANVSR_DIR/FlashVSR-v1.1"

# Check if FlashVSR directory exists
if [ ! -d "$FLASHVSR_DIR" ]; then
    echo "Error: FlashVSR directory not found at $FLASHVSR_DIR"
    echo "Please ensure FlashVSR repository is cloned in the workspace."
    exit 1
fi

# Create WanVSR directory if it doesn't exist
mkdir -p "$WANVSR_DIR"
cd "$WANVSR_DIR"

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is not installed. Installing..."

    # Try to install git-lfs
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif command -v yum &> /dev/null; then
        sudo yum install -y git-lfs
    else
        echo "Please install Git LFS manually:"
        echo "https://git-lfs.github.com/"
        exit 1
    fi
fi

# Initialize Git LFS
git lfs install

# Download model if not already downloaded
if [ -d "$MODEL_DIR" ]; then
    echo "Model directory already exists at $MODEL_DIR"
    echo "Checking for updates..."
    cd "$MODEL_DIR"
    git pull
else
    echo "Downloading FlashVSR v1.1 model from HuggingFace..."
    echo "This may take several minutes depending on your connection."
    echo ""

    # Clone with Git LFS
    git clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
    cd FlashVSR-v1.1
fi

# Verify model files
echo ""
echo "Verifying model files..."

REQUIRED_FILES=(
    "LQ_proj_in.ckpt"
    "TCDecoder.ckpt"
    "Wan2.1_VAE.pth"
    "diffusion_pytorch_model_streaming_dmd.safetensors"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
        ALL_PRESENT=false
    fi
done

echo ""
if [ "$ALL_PRESENT" = true ]; then
    echo "========================================="
    echo "✓ All model files downloaded successfully!"
    echo "Model location: $MODEL_DIR"
    echo "========================================="
else
    echo "========================================="
    echo "✗ Some model files are missing."
    echo "Please check your internet connection and try again."
    echo "========================================="
    exit 1
fi
