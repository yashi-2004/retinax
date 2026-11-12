"""RetinaX: A Research-Driven Framework for Robust and Trustworthy Retinal Image Intelligence."""

import os
from pathlib import Path

# Disable non-essential background version/telemetry network pings for offline CPU stability
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Ensure Matplotlib has a persistent writable cache directory even inside restricted sandboxes
if "MPLCONFIGDIR" not in os.environ:
    _default_mpl = Path.home() / ".matplotlib"
    _writable = False
    try:
        _default_mpl.mkdir(parents=True, exist_ok=True)
        _probe = _default_mpl / ".retinax_write_probe"
        _probe.touch()
        _probe.unlink(missing_ok=True)
        _writable = True
    except OSError:
        _writable = False
    if not _writable:
        _fallback_mpl = Path("/tmp/retinax_mpl")
        _fallback_mpl.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_fallback_mpl)

__version__ = "0.1.0"
