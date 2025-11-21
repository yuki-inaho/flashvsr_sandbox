"""
pytest configuration and fixtures for FlashVSR tests
"""
import os
import sys
from pathlib import Path

import pytest
import torch

# Add FlashVSR to Python path
WORKSPACE_ROOT = Path(__file__).parent.parent
FLASHVSR_DIR = WORKSPACE_ROOT / "FlashVSR"
sys.path.insert(0, str(FLASHVSR_DIR))


@pytest.fixture(scope="session")
def workspace_root():
    """Workspace root directory."""
    return WORKSPACE_ROOT


@pytest.fixture(scope="session")
def flashvsr_dir():
    """FlashVSR repository directory."""
    return FLASHVSR_DIR


@pytest.fixture(scope="session")
def model_dir():
    """FlashVSR v1.1 model directory."""
    return FLASHVSR_DIR / "examples" / "WanVSR" / "FlashVSR-v1.1"


@pytest.fixture(scope="session")
def sample_videos_dir():
    """Sample videos directory."""
    return FLASHVSR_DIR / "examples" / "WanVSR" / "inputs"


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def gpu_memory_gb():
    """Get GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0


@pytest.fixture(scope="session")
def block_sparse_attention_available():
    """Check if Block-Sparse Attention is available."""
    try:
        import block_sparse_attn
        return True
    except ImportError:
        return False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "model: marks tests that require downloaded models"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add gpu marker to tests that require GPU
        if "cuda_available" in item.fixturenames or "test_gpu" in item.name:
            item.add_marker(pytest.mark.gpu)

        # Add model marker to tests that require models
        if "model_dir" in item.fixturenames or "test_model" in item.name or "test_inference" in item.name:
            item.add_marker(pytest.mark.model)
