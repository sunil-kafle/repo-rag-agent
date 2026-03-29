# Make the project root importable during pytest runs.
# This allows tests to import from app/ and src/ without install-packaging first.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))