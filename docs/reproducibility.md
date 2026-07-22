# Reproducibility tools

This directory contains cluster-neutral plumbing shared by the four solvers.
It does not contain checkpoints, NRPy solution binaries, plots, or paper
results.

## Reference metadata

An NRPy solution header records grid geometry but not every Bowen-York source
parameter. Every comparison therefore requires an authenticated TOML stored in
the same directory as the supplied binary:

```bash
python -m pinndorama.reproducibility.create_reference \
  --binary /path/to/NRPYELL_solution.bin \
  --solver-config configs/punctures_3d/C04.toml \
  --output /path/to/NRPYELL_solution.reference.toml
```

For a fixed member of the parametric family, additionally pass raw Bowen-York
`--equal-spin-sz VALUE`. The generated TOML records the binary SHA-256,
SinhSymTP geometry, and complete physical source configuration. It is rejected
if it is moved away from its binary, if the digest changes, if its geometry
differs from the geometry encoded in the `NRPYELL3` header, or if the supplied
solver configuration describes a different problem.

## Canonical comparison

Build the only active interpolation reader:

```bash
make -C reference_solvers/nrpyelliptic/READER_nrpyelliptic_conformally_flat
```

Evaluate an explicit user-supplied checkpoint through the shared dispatcher.
For example, the 3D form consumes three-column native `xx0 xx1 xx2`
coordinates (or an NPZ containing exactly those three named arrays):

```bash
python -m pinndorama.reproducibility.evaluate punctures-3d \
  --config configs/punctures_3d/C04.toml \
  --checkpoint /path/to/run/checkpoint_final.npz \
  --coords /path/to/native_xx.txt \
  --output /path/to/checkpoint_values.txt
```

The `punctures-2d` form uses `configs/punctures_2d/C002_w40d4.toml`
and two-column native coordinates. The `punctures-2d-parametric` form uses
`configs/punctures_2d_parametric/P001_w40d4.toml` and additionally requires
`--equal-spin-sz VALUE`. All evaluator output tables end in the `u_nn` column.

Supply the corresponding Cartesian points to the NRPy reader and comparison
command, preserving the same row order:

```bash
python -m pinndorama.reproducibility.compare \
  --nrpy-binary /path/to/NRPYELL_solution.bin \
  --reference-config /path/to/NRPYELL_solution.reference.toml \
  --solver-config configs/punctures_3d/C04.toml \
  --coords /path/to/cartesian_xyz.txt \
  --nn-values /path/to/checkpoint_values.txt \
  --volume-weights /path/to/quadrature_weights.txt \
  --output /path/to/comparison.json
```

The reader always uses the fixed tensor-product 9-point stencil. The reported
metric is

```text
sqrt(sum_i w_i (u_NN_i - u_NRPy_i)^2 / sum_i w_i u_NRPy_i^2).
```

Weights are explicit so structured native grids and deliberately selected
point sets cannot be mixed silently. Plotting is intentionally downstream of
these explicit files; there are no cluster paths or bundled paper artifacts.

See [gpu-jax.md](gpu-jax.md) before installing a GPU-enabled JAX wheel.
