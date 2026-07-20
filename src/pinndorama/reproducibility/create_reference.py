"""Create an authenticated reference TOML beside an NRPy solution binary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from .reference import load_binary_geometry, load_solver_problem, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--solver-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--equal-spin-sz",
        type=float,
        help="Raw Bowen-York S_z for a fixed member of the parametric family.",
    )
    return parser.parse_args()


def render_reference(
    binary: Path,
    geometry: dict[str, float],
    physics: dict[str, float],
) -> str:
    lines = [
        "schema_version = 1\n",
        f'binary_file = "{binary.name}"\n',
        f'binary_sha256 = "{sha256_file(binary)}"\n',
        "\n[geometry]\n",
    ]
    lines.extend(f"{key} = {value!r}\n" for key, value in geometry.items())
    lines.append("\n[physics]\n")
    lines.extend(f"{key} = {value!r}\n" for key, value in physics.items())
    return "".join(lines)


def atomic_write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing reference TOML: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    binary = args.binary.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.parent != binary.parent:
        raise ValueError("--output must place the reference TOML beside --binary")
    geometry, physics = load_solver_problem(
        args.solver_config, equal_spin_sz=args.equal_spin_sz
    )
    encoded_geometry = load_binary_geometry(binary)
    if geometry != encoded_geometry:
        raise ValueError(
            "solver configuration geometry differs from the supplied "
            f"NRPYELL3 binary: config={geometry}, binary={encoded_geometry}"
        )
    atomic_write_new(output, render_reference(binary, geometry, physics))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
