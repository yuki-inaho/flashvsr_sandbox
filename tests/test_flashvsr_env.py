import importlib
import sys
from pathlib import Path

import pytest


def test_add_paths_makes_diffsynth_importable(monkeypatch):
    # Simulate a clean sys.path head
    monkeypatch.setattr(sys, "path", sys.path.copy())

    # Import helper and add paths
    mod = importlib.import_module("scripts.flashvsr_env")
    root, flashvsr_dir = mod.add_paths()

    # Paths should be pathlib.Path
    assert isinstance(root, Path)
    assert isinstance(flashvsr_dir, Path)

    # Paths should be present in sys.path as strings
    head = sys.path[:5]
    assert flashvsr_dir.as_posix() in head
    assert root.as_posix() in head

    # Now diffsynth should be importable
    diffsynth = importlib.import_module("diffsynth")
    assert hasattr(diffsynth, "models")

