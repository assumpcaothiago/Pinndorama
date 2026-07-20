"""Load and authenticate the repository's shared SSBroyden runtime."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any, Callable

from ._paths import SSBROYDEN_ROOT

OPTIMISTIX_ROOT = SSBROYDEN_ROOT / "optimistix"
EXPECTED_OPTIMISTIX_VERSION = "0.1.0"


def _prepend(path: Path) -> None:
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def _require_source(module: Any, *, label: str, root: Path) -> None:
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(root.resolve()):
        raise ImportError(f"vendored {label} came from {source}, not {root}")


def load_ssbroyden_api() -> tuple[type[Any], Callable[..., Any]]:
    """Return the shared solver and wrapper after authenticating their paths."""

    _prepend(SSBROYDEN_ROOT)
    _prepend(OPTIMISTIX_ROOT)
    importlib.invalidate_caches()

    optimistix = importlib.import_module("optimistix")
    _require_source(optimistix, label="Optimistix", root=OPTIMISTIX_ROOT)
    if optimistix.__version__ != EXPECTED_OPTIMISTIX_VERSION:
        raise ImportError(
            "vendored Optimistix version mismatch: expected "
            f"{EXPECTED_OPTIMISTIX_VERSION}, loaded {optimistix.__version__}"
        )

    wrapper = importlib.import_module("optimistix_wrapper")
    family = importlib.import_module("ssbroyden_family")
    _require_source(wrapper, label="Optimistix wrapper", root=SSBROYDEN_ROOT)
    _require_source(family, label="SSBroyden family", root=SSBROYDEN_ROOT)
    return family.SSBroyden, wrapper.run_optimization
