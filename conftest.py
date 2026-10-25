"""Repository-root conftest so `pytest` works from any directory.

Streamlit's ``AppTest`` imports the app script by path, and the app itself
imports project packages (``agents``, ``tools``, ...). Guaranteeing the project
root on ``sys.path`` makes `pytest` and `python -m pytest` behave identically.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
