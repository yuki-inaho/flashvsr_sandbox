"""
Environment tests for FlashVSR setup
Tests CUDA, PyTorch, Block-Sparse Attention, and model files
"""
import sys
from pathlib import Path

import pytest
import torch


class TestCUDAEnvironment:
    """Test CUDA environment setup."""

    def test_cuda_available(self):
        """Test if CUDA is available."""
        assert torch.cuda.is_available(), "CUDA is not available"

    def test_cuda_version(self):
        """Test CUDA version compatibility."""
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")
            # CUDA 11.8 or later is expected
            assert cuda_version is not None, "CUDA version is None"

    def test_gpu_count(self):
        """Test GPU availability."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")
            assert gpu_count > 0, "No GPUs found"

    def test_gpu_memory(self, gpu_memory_gb):
        """Test GPU memory capacity."""
        if torch.cuda.is_available():
            print(f"GPU memory: {gpu_memory_gb:.2f} GB")
            # FlashVSR might require significant memory
            # This is just informational, not a strict requirement
            if gpu_memory_gb < 8:
                pytest.skip(f"GPU memory ({gpu_memory_gb:.2f} GB) may be insufficient")

    def test_gpu_device_name(self):
        """Test GPU device name."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU device: {device_name}")
            assert device_name, "GPU device name is empty"


class TestPyTorchEnvironment:
    """Test PyTorch environment setup."""

    def test_pytorch_version(self):
        """Test PyTorch version."""
        print(f"PyTorch version: {torch.__version__}")
        # Expecting torch 2.6.0
        major, minor = map(int, torch.__version__.split('.')[:2])
        assert (major, minor) >= (2, 6), f"PyTorch version {torch.__version__} is too old"

    def test_bfloat16_support(self):
        """Test bfloat16 support (required for FlashVSR)."""
        if torch.cuda.is_available():
            # FlashVSR uses bfloat16
            device = torch.device("cuda:0")
            try:
                x = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
                assert x.dtype == torch.bfloat16
                print("✓ bfloat16 is supported")
            except Exception as e:
                pytest.fail(f"bfloat16 not supported: {e}")


class TestBlockSparseAttention:
    """Test Block-Sparse Attention installation."""

    def test_block_sparse_attention_import(self, block_sparse_attention_available):
        """Test if Block-Sparse Attention can be imported."""
        if not block_sparse_attention_available:
            pytest.skip("Block-Sparse Attention is not installed")

        import block_sparse_attn
        print(f"✓ Block-Sparse Attention imported successfully")

    def test_block_sparse_attention_version(self, block_sparse_attention_available):
        """Test Block-Sparse Attention version."""
        if not block_sparse_attention_available:
            pytest.skip("Block-Sparse Attention is not installed")

        import block_sparse_attn
        if hasattr(block_sparse_attn, "__version__"):
            print(f"Block-Sparse Attention version: {block_sparse_attn.__version__}")


class TestFlashVSRRepository:
    """Test FlashVSR repository structure."""

    def test_flashvsr_dir_exists(self, flashvsr_dir):
        """Test if FlashVSR directory exists."""
        assert flashvsr_dir.exists(), f"FlashVSR directory not found: {flashvsr_dir}"

    def test_flashvsr_examples_exist(self, flashvsr_dir):
        """Test if FlashVSR examples directory exists."""
        examples_dir = flashvsr_dir / "examples" / "WanVSR"
        assert examples_dir.exists(), f"Examples directory not found: {examples_dir}"

    def test_sample_videos_exist(self, sample_videos_dir):
        """Test if sample videos exist."""
        assert sample_videos_dir.exists(), f"Sample videos directory not found: {sample_videos_dir}"

        # Check for at least one sample video
        sample_videos = list(sample_videos_dir.glob("*.mp4"))
        assert len(sample_videos) > 0, "No sample videos found"
        print(f"Found {len(sample_videos)} sample videos")


@pytest.mark.model
class TestModelFiles:
    """Test FlashVSR v1.1 model files."""

    def test_model_dir_exists(self, model_dir):
        """Test if model directory exists."""
        if not model_dir.exists():
            pytest.skip(f"Model directory not found: {model_dir}\nPlease run: bash scripts/download_model_v1.1.sh")

        assert model_dir.exists(), f"Model directory not found: {model_dir}"

    def test_required_model_files(self, model_dir):
        """Test if all required model files exist."""
        if not model_dir.exists():
            pytest.skip(f"Model directory not found: {model_dir}")

        required_files = [
            "LQ_proj_in.ckpt",
            "TCDecoder.ckpt",
            "Wan2.1_VAE.pth",
            "diffusion_pytorch_model_streaming_dmd.safetensors",
        ]

        missing_files = []
        for filename in required_files:
            file_path = model_dir / filename
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / 1024**2
                print(f"✓ {filename} ({file_size_mb:.2f} MB)")
            else:
                missing_files.append(filename)
                print(f"✗ {filename} (missing)")

        if missing_files:
            pytest.fail(f"Missing model files: {', '.join(missing_files)}\nPlease run: bash scripts/download_model_v1.1.sh")


class TestFlashVSRImports:
    """Test FlashVSR module imports."""

    def test_flashvsr_path_setup(self, flashvsr_dir):
        """Test if FlashVSR is in Python path."""
        assert str(flashvsr_dir) in sys.path or any(
            str(flashvsr_dir) in p for p in sys.path
        ), "FlashVSR directory not in Python path"

    @pytest.mark.model
    def test_import_flashvsr_pipeline(self, model_dir):
        """Test importing FlashVSR pipeline."""
        if not model_dir.exists():
            pytest.skip("Model directory not found")

        try:
            # Try to import FlashVSR components
            # Note: This may fail if the module structure is different
            # Adjust based on actual FlashVSR structure
            print("Attempting to import FlashVSR components...")
            # from examples.WanVSR.models import FlashVSRTinyPipeline
            print("✓ Import test placeholder (adjust based on actual structure)")
        except ImportError as e:
            print(f"Import test skipped: {e}")
            pytest.skip(f"Cannot import FlashVSR components: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
