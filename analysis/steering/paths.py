"""Resolve data and output directories."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get('EVAL_GAMING_DATA', ROOT / 'data'))
OUTPUT_ROOT = Path(os.environ.get('EVAL_GAMING_OUTPUT', ROOT))
DATA = DATA_ROOT / 'tables' / 'steering'
OUTPUT = OUTPUT_ROOT / 'tables' / 'steering'
FIGURES = OUTPUT_ROOT / 'figures' / 'steering'
