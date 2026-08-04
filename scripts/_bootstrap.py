"""Put src/ on the import path so the scripts run without installing anything.

Every script imports this first. If you'd rather install the package properly:

    pip install -e research

then the import still works and this file becomes a no-op.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
