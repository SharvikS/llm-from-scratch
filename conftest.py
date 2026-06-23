"""Ensure the repo root is importable during test collection.

Lets tests use absolute imports like `from model.transformer import Transformer`
no matter which directory pytest is invoked from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
