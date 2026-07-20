# Parametric 2D SinhSymTP PINN

This is the publication-facing equal-spin solver. Its third network input,
`equal_spin_sz`, is the raw Bowen-York component `S_z`; every source evaluation
sets `S0_z = S1_z = equal_spin_sz`. P001 samples that input at 80 cell centers
over `[-0.2, 0.2]`.

The float64 `w40d4` network consumes `[xx0, xx1, equal_spin_sz]`. Its output is

```text
u = raw_network_output / sqrt(1 + r_sph^2).
```

Training minimizes the regularized Hamiltonian residual `J R_H`, lower/upper
theta scalar-parity terms, and the scaled outer Robin condition. Adam is a
warm-up for the same full loss subsequently refined by the vendored SSBroyden
implementation.

## Train and continue

```bash
python -m pinndorama.solvers.punctures_2d_parametric.train \
  --config configs/punctures_2d_parametric/P001_w40d4.toml \
  --output-dir /path/to/run

python -m pinndorama.solvers.punctures_2d_parametric.train \
  --config configs/punctures_2d_parametric/P001_w40d4.toml \
  --output-dir /path/to/continued-run \
  --resume-checkpoint /path/to/run/checkpoint_final.npz
```

The only optional overrides are `--Nxx0`, `--Nxx1`,
`--equal-spin-sz-points`, `--seed`, `--adam-steps`, and
`--ssbroyden-steps`. Physics, geometry, loss, architecture, and output ansatz
are controlled by the checked-in TOML file.

`Nxx0` and `Nxx1` count native-coordinate PINN collocation samples; they do
not include finite-difference ghost zones. `equal_spin_sz_points` separately
counts samples along the physical parameter axis and is not `Nxx2`.

Checkpoints are atomic `.npz` files with ten float64 arrays (`leaf_000` through
`leaf_009`) alternating Equinox-layout dense weights and biases, plus UTF-8 JSON
metadata stored as `uint8` under `metadata_json`. Continuing a run validates
the immutable physical and model fields and records the SHA-256 hash of the
parent checkpoint. Continuation deliberately initializes a fresh optimizer,
so optimizer and collocation settings may change. Pickle checkpoints are
intentionally unsupported.

## Evaluate

Supply a text file containing native `xx0 xx1` columns (or an NPZ containing
exactly the arrays `xx0` and `xx1`):

```bash
python -m pinndorama.solvers.punctures_2d_parametric.evaluate \
  --config configs/punctures_2d_parametric/P001_w40d4.toml \
  --checkpoint /path/to/checkpoint_final.npz \
  --equal-spin-sz 0.1 \
  --coords /path/to/native_coords.txt \
  --output /path/to/prediction.txt
```

The evaluator checks both checkpoint/config compatibility and the declared
equal-spin range. It never searches for checkpoints or reference data.

## Source expressions

`nrpy_expression_builder.py` constructs the axisymmetric SinhSymTP expressions
from the repository's pinned upstream `vendor/nrpy/` snapshot. The fixed source fields
are the two bare masses, `zPunc`, momenta, and transverse spins. The two axial
spin components are supplied only through `equal_spin_sz`.

The shared SSBroyden runtime and its Optimistix dependency are documented in
`vendor/ssbroyden/VENDORED_FROM.md`.
