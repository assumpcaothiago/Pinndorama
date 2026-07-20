# NRPyElliptic reference applications

- `axisymmetric/` contains the standard axisymmetric reference solver.
- `three_dimensional/` contains the standard 3D solver, the canonical fixed
  9-point comparison reader, and the provenance-checked parametric-neural-
  network warm-start variant.

Each C application builds from its own directory with `make`. Solution binaries
remain user-supplied and require accompanying authenticated reference TOML
metadata for publication comparisons.
