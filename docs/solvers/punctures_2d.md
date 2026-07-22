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

Compare a directory of final checkpoints with an authenticated three-dimensional
NRPyElliptic reference on one independent cell-centered grid:

```bash
OMP_NUM_THREADS=8 python -m pinndorama.solvers.punctures_2d.relative_l2_nrpy \
  --config configs/punctures_2d/C004_square_w40d4.toml \
  --grid-mode common \
  --Nxx0 760 \
  --Nxx1 760 \
  --radius 7.5 \
  --output /path/to/relative_l2_common.tsv \
  /path/to/checkpoint/root
```

Use `--grid-mode native` to evaluate each checkpoint on its own training grid.
In either mode, points from every required unique grid are concatenated and sent
to the canonical nine-point reader in one invocation. The native-grid reference
values are then reused across checkpoints with matching resolutions. The metric
is the volume-weighted relative `L2` norm, with weights
`sqrt(det(gamma_hat))*dxx0*dxx1`, restricted by default to `r<7.5M`.
Checkpoint inference uses the equivalent NumPy forward pass, so this
postprocessing command does not initialize JAX or require a GPU.

The default binary is
`reference_solvers/nrpyelliptic/nrpyelliptic_conformally_flat_symmetricID/NRPYELL_solution.bin`.
Its authenticated `NRPYELL_solution.reference.toml` must be stored beside it.
Create that metadata after generating the binary:

```bash
python -m pinndorama.reproducibility.create_reference \
  --binary reference_solvers/nrpyelliptic/nrpyelliptic_conformally_flat_symmetricID/NRPYELL_solution.bin \
  --solver-config configs/punctures_2d/C004_square_w40d4.toml \
  --output reference_solvers/nrpyelliptic/nrpyelliptic_conformally_flat_symmetricID/NRPYELL_solution.reference.toml
```

The binary, authenticated reference metadata, interpolation scratch files, and
result TSV are analysis artifacts and are not part of the source distribution.

Generate seed-resolved and aggregate convergence figures from a common-grid
comparison TSV:

```bash
python -m pinndorama.solvers.punctures_2d.plot_relative_l2_convergence \
  --input /path/to/relative_l2_common.tsv \
  --output-prefix /path/to/c004_convergence
```

The plotting command requires one coherent common evaluation grid and rejects
mixed reference hashes, radii, sample counts, or incomplete seed sets. It writes
seed-curve and median/interquartile figures as PDF and PNG, together with a
per-resolution summary TSV and a JSON provenance record. These generated files
are analysis artifacts and remain outside the source repository.

The three configs capture C002 `w40d4`, C002 `w60d10`, and the C004 square-grid
`w40d4` anchor. They contain only the upstream `zPunc` separation model.
