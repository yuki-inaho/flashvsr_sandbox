"""
Inference tests for FlashVSR v1.1
Tests basic inference functionality with sample videos
"""
import sys
from pathlib import Path

import pytest
import torch
import imageio


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
class TestFlashVSRInference:
    """Test FlashVSR inference with sample videos."""

    @pytest.fixture(scope="class")
    def sample_video(self, sample_videos_dir):
        """Get the smallest sample video for testing."""
        videos = sorted(sample_videos_dir.glob("*.mp4"), key=lambda p: p.stat().st_size)
        if not videos:
            pytest.skip("No sample videos found")
        # Use the smallest video for testing
        smallest_video = videos[0]
        print(f"Using sample video: {smallest_video.name} ({smallest_video.stat().st_size / 1024**2:.2f} MB)")
        return smallest_video

    def test_load_video(self, sample_video):
        """Test loading a video with imageio."""
        try:
            reader = imageio.get_reader(str(sample_video))
            metadata = reader.get_meta_data()
            print(f"Video metadata: {metadata}")

            # Read first frame
            first_frame = reader.get_data(0)
            print(f"First frame shape: {first_frame.shape}")

            assert first_frame is not None
            assert len(first_frame.shape) == 3  # H, W, C
            reader.close()
        except Exception as e:
            pytest.fail(f"Failed to load video: {e}")

    def test_video_to_tensor(self, sample_video):
        """Test converting video to tensor."""
        try:
            reader = imageio.get_reader(str(sample_video))
            frames = []
            for i, frame in enumerate(reader):
                if i >= 8:  # Read first 8 frames for testing
                    break
                frames.append(frame)
            reader.close()

            # Convert to tensor
            frames_tensor = torch.from_numpy(frames[0]).float()
            print(f"Frame tensor shape: {frames_tensor.shape}")
            print(f"Frame tensor dtype: {frames_tensor.dtype}")

            assert frames_tensor.shape[2] == 3  # RGB channels
        except Exception as e:
            pytest.fail(f"Failed to convert video to tensor: {e}")

    @pytest.mark.skip(reason="Requires complete FlashVSR pipeline setup and significant GPU memory")
    def test_flashvsr_inference_tiny(self, sample_video, model_dir, cuda_available):
        """Test FlashVSR Tiny inference (memory-efficient version)."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # This test is skipped by default as it requires:
        # 1. Proper FlashVSR pipeline initialization
        # 2. Significant GPU memory
        # 3. Block-Sparse Attention working correctly

        # Example placeholder for actual inference code
        # Adjust based on FlashVSR's actual API
        try:
            # from examples.WanVSR.models import FlashVSRTinyPipeline

            # Load model
            # pipe = FlashVSRTinyPipeline(...)

            # Run inference
            # result = pipe(...)

            pytest.skip("Inference test placeholder - implement based on actual API")
        except Exception as e:
            pytest.fail(f"Inference failed: {e}")

    def test_memory_requirements(self, cuda_available, gpu_memory_gb):
        """Test and report GPU memory requirements."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        print(f"\nGPU Memory Information:")
        print(f"  Total memory: {gpu_memory_gb:.2f} GB")

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  Currently allocated: {allocated:.4f} GB")
        print(f"  Currently reserved: {reserved:.4f} GB")

        # Create a small tensor to test memory allocation
        try:
            test_size = 1024 * 1024 * 100  # 100M floats ~= 400MB
            test_tensor = torch.randn(test_size, dtype=torch.float32, device="cuda:0")

            allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  After test allocation: {allocated_after:.4f} GB")

            del test_tensor
            torch.cuda.empty_cache()

            print(f"  Available for inference: ~{gpu_memory_gb - allocated:.2f} GB")

            # Rough estimate: FlashVSR Tiny might need 8-12 GB for moderate resolution
            if gpu_memory_gb < 8:
                pytest.skip(f"GPU memory ({gpu_memory_gb:.2f} GB) likely insufficient for FlashVSR")

        except RuntimeError as e:
            pytest.fail(f"Memory allocation test failed: {e}")


@pytest.mark.gpu
class TestGPUPerformance:
    """Test GPU performance characteristics."""

    def test_torch_compile_available(self):
        """Test if torch.compile is available (PyTorch 2.0+)."""
        assert hasattr(torch, "compile"), "torch.compile not available"
        print("✓ torch.compile is available")

    def test_gpu_tensor_operations(self):
        """Test basic GPU tensor operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda:0")

        # Test basic operations
        x = torch.randn(1000, 1000, device=device, dtype=torch.float32)
        y = torch.randn(1000, 1000, device=device, dtype=torch.float32)

        # Matrix multiplication
        z = torch.matmul(x, y)
        assert z.device.type == "cuda"

        # Test bfloat16
        x_bf16 = x.to(torch.bfloat16)
        y_bf16 = y.to(torch.bfloat16)
        z_bf16 = torch.matmul(x_bf16, y_bf16)

        assert z_bf16.dtype == torch.bfloat16
        print("✓ GPU tensor operations working correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
