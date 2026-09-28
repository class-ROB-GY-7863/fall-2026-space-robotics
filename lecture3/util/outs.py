from datetime import datetime
from pathlib import Path

OUTS = Path(__file__).parent.parent / "outs"


def make_out_dir(name):
    """Create and return outs/<YYYYmmdd_HHMMSS>_<name>/."""
    out = OUTS / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    return out
