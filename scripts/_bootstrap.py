"""Add the source checkout to the import path for CLI scripts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "third_party/holosoma", ROOT):
    sys.path.insert(0, str(directory))
