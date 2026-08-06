"""Put src/ on the import path so the scripts run without installing anything.

Every script imports this first. After `poetry install` the package is already
importable and this becomes a no-op -- it is kept so the scripts still work from
a bare python, e.g. on a machine where you haven't set the environment up yet.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
