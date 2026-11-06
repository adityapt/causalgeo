"""Check the saved manuscript results against the published rounded numbers.

The full Monte Carlo rebuild is paper/reproduce.py. This test runs only the
comparison, so a change to the saved tables fails here without refitting.
"""

import importlib.util
import sys
from pathlib import Path


def _reproduce():
    paper = Path(__file__).resolve().parents[1] / "paper"
    if str(paper) not in sys.path:
        sys.path.insert(0, str(paper))
    path = paper / "reproduce.py"
    spec = importlib.util.spec_from_file_location("manuscript_reproduce", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_results_match_the_published_numbers():
    _reproduce()._check_tables()
