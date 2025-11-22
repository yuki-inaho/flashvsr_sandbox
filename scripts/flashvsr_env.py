"""
Path-based helper to make FlashVSR modules importable when running pytest or
ad-hoc scripts from the repo root.

Usage
-----
- Near the top of your script/test:
    from scripts.flashvsr_env import add_paths
    add_paths()
- Or run directly to inspect the resolved paths:
    python scripts/flashvsr_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple


def add_paths() -> Tuple[Path, Path]:
    """
    Insert repo root and FlashVSR package directory into sys.path (if absent).
    Returns (root, flashvsr_dir) as Path objects.
    """
    root = Path(__file__).resolve().parents[1]
    flashvsr_dir = root / "FlashVSR"

    for path in (flashvsr_dir, root):
        path_str = path.as_posix()
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root, flashvsr_dir


if __name__ == "__main__":
    root, flashvsr_dir = add_paths()
    print(f"Added to sys.path:\n  FlashVSR: {flashvsr_dir}\n  Root:     {root}")
    print("Current sys.path (head):")
    for p in sys.path[:10]:
        print(f"  {p}")
