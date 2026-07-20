# Publication 3D SinhSymTP PINN

This directory contains the publication-facing float64 JAX solver for the
three-dimensional conformally flat Hamiltonian constraint in native SinhSymTP
coordinates.

The retained objective is

```text
loss = mean((J R_H)^2)
     + theta-axis regularity
     + phi periodicity
     + scaled outer Robin
```

`J R_H` is evaluated directly in divergence form. The network is a `tanh` MLP
on `(xx0, xx1, xx2)` with the fixed smooth ansatz
`raw_MLP / sqrt(1 + r^2)`. The source is the upstream NRPy Bowen--York
two-puncture model with punctures separated only along the z axis.

## Reproducing the publication stages

Fresh C04 training:

```bash
python -m pinndorama.solvers.punctures_3d.train \
  --config configs/punctures_3d/C04.toml \
  --output-dir runs/C04
```

C05 is a continuation with two-zone `xx0` sampling and therefore requires the
C04 checkpoint explicitly:

```bash
python -m pinndorama.solvers.punctures_3d.train \
  --config configs/punctures_3d/C05.toml \
  --output-dir runs/C05 \
  --resume-checkpoint runs/C04/checkpoint_final.npz
```

Only collocation resolution, seed, Adam length, and SSBroyden length may be
overridden on the command line. `Nxx0`, `Nxx1`, and `Nxx2` count
cell-centered native-coordinate PINN samples and do not include
finite-difference ghost zones; their overrides are `--Nxx0`, `--Nxx1`, and
`--Nxx2`. Geometry, physics, architecture, ansatz, loss, sampling rule, and
optimizer tolerances live in TOML and are validated before training.

## Checkpoints and evaluation

Checkpoints are atomic NPZ files. Parameters use the stable order
`leaf_000=W0, leaf_001=b0, ...`, with dense weights stored in Equinox
`(out_features, in_features)` order; `metadata_json` is a UTF-8 JSON document
stored as a one-dimensional `uint8` array. No pickle is used. A continuation
validates the checkpoint's dtype, architecture, geometry, physics, and ansatz
and records the SHA256 of its parent checkpoint.

Evaluate on explicit native-coordinate arrays (`xx0`, `xx1`, `xx2`):

```bash
python -m pinndorama.solvers.punctures_3d.evaluate \
  --config configs/punctures_3d/C04.toml \
  --checkpoint runs/C04/checkpoint_final.npz \
  --coords native_xx_points.txt \
  --output solution.txt
```

Coordinates may instead be an NPZ containing exactly `xx0`, `xx1`, and `xx2`.
The evaluator validates the checkpoint against the supplied scientific TOML
and writes `xx0 xx1 xx2 x y z u_nn`; it has no cluster paths or scheduler
assumptions.

The runtime environment must provide NumPy, SymPy, JAX with float64 enabled,
Optax, Equinox, Lineax, JAXTyping, and Typing Extensions. The vendored
Optimistix `pyproject.toml` records its compatibility bounds; accelerator- and
cluster-specific JAX installation belongs to the external workflow.

## Optimizer provenance

The shared SSBroyden implementation is vendored in `vendor/ssbroyden/`. Its
historical source gitlink was
`4c87785c68f0fec6b09000f474daef76fb181eea`; the accepted-state loss reuse is
the local patch recorded by Pinndorama commit `0010580`. See the vendored
README and `VENDORED_FROM.md` for citations and the nested Optimistix Apache-2.0
license. No license is asserted here for the add-on files beyond the licenses
provided by their original authors.
