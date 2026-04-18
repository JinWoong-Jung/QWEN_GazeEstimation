from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from model.xxx import ...` works
# when pytest is invoked from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
