import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github/scripts"))
spec = importlib.util.spec_from_file_location(
    "wheel_index", ROOT / ".github/scripts/generate-wheel-index.py"
)
wheel_index = importlib.util.module_from_spec(spec)
sys.modules["wheel_index"] = wheel_index
spec.loader.exec_module(wheel_index)
