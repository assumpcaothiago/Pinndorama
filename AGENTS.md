# AGENTS.md

## Purpose

Maintain Pinndorama as a reproducible research codebase for the spherical 1D
toy problem, the fixed and parametric 2D solvers, the full 3D solver, and their
NRPyElliptic comparisons. Keep scientific changes explicit: state when an edit
changes a residual, coordinate map, collocation rule, boundary term, model
ansatz, optimizer stage, checkpoint contract, or interpolation convention.

## Repository responsibilities

- `src/pinndorama/solvers/` contains four independent JAX solver packages:
  `toy_problem_1d`, `punctures_2d`, `punctures_2d_parametric`, and
  `punctures_3d`.
- `src/pinndorama/reproducibility/` contains checkpoint evaluation, NRPy
  reference metadata, interpolation orchestration, and comparison metrics.
- `configs/` contains explicit TOML inputs grouped by solver name.
- `reference_solvers/nrpyelliptic/` contains the standalone axisymmetric and
  3D C applications, the canonical reader, and the parametric-network warm
  start.
- `vendor/nrpy/` and `vendor/ssbroyden/` contain pinned third-party source.
- `docs/` explains solver-specific and comparison workflows; `tests/` defines
  the acceptance behavior.

Use package-relative imports within `pinndorama`. Resolve repository resources
through `src/pinndorama/_paths.py`; do not reconstruct vendor, configuration,
reader, or reference-solver paths in individual modules. Keep the solver
implementations separate unless a change explicitly introduces and validates a
shared abstraction.

## Scientific contract

- Enable JAX x64 and preserve float64 model parameters, calculations, and
  checkpoint arrays.
- For `toy_problem_1d`, preserve the NRPy SinhSpherical map, configured
  non-parametric mass `m`, the regularized residual `r R`, explicit
  `u_r(0)=0`, and exactly one configured outer condition: the first-order
  Robin operator `(1+r)(r u_r+u)` or the second-order Robin operator
  `r^3 (u_r + u/r) - 2m`. Either may be enforced only at `xx0=1` or over the
  configured physical-radius band plus that endpoint. The manufactured exact
  solution is validation data, not a training target or network input.
- Work in native SinhSymTP coordinates and use only the pinned NRPy z-axis
  puncture-separation model.
- Train the divergence-form, Jacobian-regularized Hamiltonian residual
  `J R_H`. Do not restore raw-residual or alternative-normalization modes.
- Preserve theta parity or axis regularity, 3D phi periodicity, and the scaled
  outer Robin condition used by the puncture solvers.
- Preserve the output transform `raw / sqrt(1 + r^2)`.
- Use cell-centered interior collocation. `Nxx0`, `Nxx1`, and `Nxx2` count
  PINN samples in native coordinates; they are not finite-difference sizes and
  do not include ghost zones. The C04 run uses uniform `xx0` sampling; C05
  must resume from C04 and changes to its checked two-zone `xx0` sampling with
  Adam disabled.
- Treat `equal_spin_sz` as raw Bowen--York `S_z`, with
  `S0_z = S1_z = equal_spin_sz`. Never describe it as dimensionless spin.
- Keep Adam followed by the shared faithful SSBroyden implementation.
- Compare NRPy solutions with the canonical fixed 9-point reader and the
  volume-weighted relative L2 metric.

## Supported interfaces

Training commands follow:

```bash
python -m pinndorama.solvers.<solver>.train \
  --config CONFIG.toml --output-dir DIR \
  [--resume-checkpoint CHECKPOINT.npz]
```

Only resolution, seed, Adam length, SSBroyden length, output directory, and
resume checkpoint may be command-line overrides. Physics, geometry, loss,
architecture, dtype, ansatz, sampling policy, and optimizer tolerances belong
in TOML. Resolution overrides are spelled `--Nxx0`, `--Nxx1`, and `--Nxx2`
to match NRPy; the 1D solver accepts only `--Nxx0`, and the 2D solvers do not
accept `--Nxx2`.

Evaluation commands follow:

```bash
python -m pinndorama.reproducibility.evaluate <solver-name> \
  --config CONFIG.toml --checkpoint CHECKPOINT.npz \
  --coords COORDS --output OUTPUT
```

The evaluator names are `toy-problem-1d`, `punctures-2d`,
`punctures-2d-parametric`, and `punctures-3d`. The parametric form also
requires `--equal-spin-sz`.

Every trainer writes `OUTPUT_DIR/checkpoint_final.npz`. Keep the versioned,
atomic NPZ format: float64 parameter arrays plus JSON metadata encoded as a
`uint8` array. Resume must reject incompatible physics, geometry,
architecture, dtype, and ansatz, while allowing documented optimizer and
collocation changes and recording the parent SHA-256. Do not add pickle
compatibility.

Require explicit paths for checkpoints, coordinates, NRPy binaries, reference
TOMLs, outputs, and plots. Do not add machine-specific defaults, scheduler
directives, runtime downloads, or implicit searches for research artifacts.

## Reference solvers and vendored code

NRPy binaries require an authenticated reference TOML beside the binary
because their headers omit some physical parameters. Preserve restart
precedence and the `zero | parametric_nn` initializer behavior in the 3D warm
start. Neural initialization must use a generated, provenance-complete header.

Refresh `vendor/nrpy/` only by replacing the complete source tree with one
reviewed upstream commit, excluding version-control metadata. Update
`VENDORED_FROM.md` and rerun all three symbolic-builder tests. Keep local
Pinndorama adaptations outside the vendored NRPy tree.

All neural solvers must load the single runtime under `vendor/ssbroyden/`.
Reject ambient Optimistix shadowing and do not require separate Optimistix
distribution metadata. Preserve the nested license, provenance, algorithm
README, accepted-loss wrapper patch, and `SHA256SUMS`. Any runtime edit requires
updated provenance and checksums plus float64 SSBroyden smokes for all three
puncture solvers and the 1D toy solver. Do not assign a license to the add-on
files while written permission is pending.

## Validation

Choose checks in proportion to the change:

- Documentation-only edits: verify every path and command, run relevant
  `--help` interfaces, and run `git diff --check`.
- Python edits: compile changed modules with
  `PYTHONPYCACHEPREFIX=/tmp/pinndorama_pycache python -m py_compile ...` and
  run focused pytest targets.
- Numerical edits: run formula or sampling tests and the affected float64
  Adam-to-SSBroyden smoke tests.
- Vendored NRPy edits: run all symbolic-builder and vendor-integrity tests.
- Vendored SSBroyden edits: verify `SHA256SUMS`, import provenance, metadata
  independence, and all four optimizer smokes.
- C edits: build the affected application, exercise its help or integration
  fixture, and run `make clean`.
- Broad or cross-cutting edits: run `python -m pytest` and
  `git diff --check`.

Before handing off, confirm active files contain no cluster-specific absolute
paths, batch-scheduler directives, retired puncture parameters, pickle
checkpoints, or generated research artifacts.

## Git and generated files

Inspect `git status --short` before staging. Preserve unrelated user changes
and stage only the intended source, tests, configurations, and documentation.
Do not commit objects, executables, checkpoints, solution binaries, generated
headers, plots, tables, reports, caches, or saved models. Avoid destructive Git
commands.
