# Fixed 2D SinhSymTP solver

This is the publication-facing axisymmetric JAX solver. It trains in native
SinhSymTP coordinates with float64 arrays and the divergence-form regularized
Hamiltonian residual

```text
J R_H = d_0(J g^00 u_,0) + d_1(J g^11 u_,1)
        + J (ADD_times_AUU / 8) (psi_background + u)^(-7).
```

The total loss is the mean-square `J R_H` plus theta-parity regularity and the
scaled outer Robin condition `(1+r)(r du/dr + u)`. The network ansatz is
`u = raw / sqrt(1+r^2)`. Interior and polar coordinates are cell-centered;
parity and outer-boundary samples remain structured. Adam is followed by the
pinned full-JIT SSBroyden implementation shared under `vendor/ssbroyden/`.

The TOML fields `Nxx0` and `Nxx1` count cell-centered PINN collocation samples
in native SinhSymTP coordinates. They are not finite-difference grid sizes and
do not include ghost zones. Their optional CLI overrides are `--Nxx0` and
`--Nxx1`.

Run a checked-in paper configuration explicitly:

```bash
python -m pinndorama.solvers.punctures_2d.train \
  --config configs/punctures_2d/C002_w40d4.toml \
  --output-dir /path/to/run
```

Continuation uses only the clean-break NPZ schema:

```bash
python -m pinndorama.solvers.punctures_2d.train \
  --config configs/punctures_2d/C002_w40d4.toml \
  --output-dir /path/to/continued-run \
  --resume-checkpoint /path/to/checkpoint_final.npz
```

Immutable physics, geometry, architecture, dtype, and output ansatz are checked
on resume. Optimizer and collocation settings may change, and the child
checkpoint records its parent's SHA256. Legacy pickle checkpoints are not read.
Model leaves are stored as `leaf_000`, `leaf_001`, ... in Equinox dense-layer
weight/bias order, with weights shaped `(out_features, in_features)`.

Evaluate explicit native-coordinate points (`xx0 xx1`) without bundled data:

```bash
python -m pinndorama.solvers.punctures_2d.evaluate \
  --config configs/punctures_2d/C002_w40d4.toml \
  --checkpoint /path/to/checkpoint_final.npz \
  --coords /path/to/native_coords.txt \
  --output /path/to/predictions.txt
```

The three configs capture C002 `w40d4`, C002 `w60d10`, and the C004 square-grid
`w40d4` anchor. They contain only the upstream `zPunc` separation model.
