# Pinndorama

Pinndorama uses physics-informed neural networks (PINNs) for nonlinear elliptic
problems. It contains a spherical 1D manufactured problem and fixed
axisymmetric, equal-spin parametric, and fully three-dimensional JAX solvers
for the conformally flat Hamiltonian constraint. NRPyElliptic applications are
included for generating reference puncture solutions.

The repository provides source code, checked configurations, and comparison
tools. Trained checkpoints, NRPy solution binaries, plots, and paper results
are not bundled.

## Repository layout

- `src/pinndorama/` contains the installable Python package, including the
  four neural solvers and shared reproducibility tools.
- `configs/` contains the checked TOML files that define solver runs.
- `reference_solvers/nrpyelliptic/` contains the axisymmetric and 3D C
  reference solvers, the canonical interpolation reader, and the optional
  parametric-network warm start.
- `vendor/nrpy/` and `vendor/ssbroyden/` contain pinned source dependencies
  needed by the solvers.
- `docs/` contains detailed solver, GPU-installation, and comparison guides.
- `tests/` checks the symbolic expressions, numerical terms, checkpoints,
  optimizers, C tools, and end-to-end interfaces.

For a consolidated account of the package structure, equations, losses,
training stages, checkpoints, and vendored dependencies, see the standalone
[technical reference](docs/technical_reference.tex).

## Installation

Clone the repository and create a Python 3.12 virtual environment (Python 3.11
is also supported):

```bash
git clone <pinndorama-repository-url>
cd Pinndorama

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The editable installation makes `pinndorama` importable while keeping the
working tree as the source of truth. It also installs the ordinary Python
dependencies declared in `pyproject.toml`. SSBroyden and the required modified
Optimistix runtime are already stored under `vendor/ssbroyden/`; they do not
need a separate installation or a network download.

The default dependency set uses platform-neutral JAX. Follow
[the GPU JAX guide](docs/gpu-jax.md) when installing for an accelerator.

## Solvers and configurations

| Solver | Training module | Checked configurations | Guide |
| --- | --- | --- | --- |
| Spherical 1D toy problem | `pinndorama.solvers.toy_problem_1d.train` | `T001_w40d4` | [1D toy solver](docs/solvers/toy_problem_1d.md) |
| Fixed axisymmetric 2D | `pinndorama.solvers.punctures_2d.train` | `C002_w40d4`, `C002_w60d10`, `C004_square_w40d4` | [Fixed 2D solver](docs/solvers/punctures_2d.md) |
| Equal-spin parametric 2D | `pinndorama.solvers.punctures_2d_parametric.train` | `P001_w40d4` | [Parametric 2D solver](docs/solvers/punctures_2d_parametric.md) |
| Full 3D | `pinndorama.solvers.punctures_3d.train` | `C04`, `C05` | [3D solver](docs/solvers/punctures_3d.md) |

All training runs are driven by an explicit TOML file. The command line may
override resolution, random seed, stage lengths, output location, and resume
checkpoint; the physical problem, geometry, loss, architecture, and output
ansatz remain in the configuration.

Resolution follows NRPy's native-coordinate notation: `Nxx0` and `Nxx1` are
the cell-centered collocation counts for the two axisymmetric coordinates, and
3D configurations additionally use `Nxx2`. These are PINN sample counts, not
finite-difference dimensions, and they do not include ghost zones. The matching
command-line overrides are `--Nxx0`, `--Nxx1`, and, for 3D, `--Nxx2`.

### Spherical 1D toy problem

```bash
python -m pinndorama.solvers.toy_problem_1d.train \
  --config configs/toy_problem_1d/T001_w40d4.toml \
  --output-dir runs/T001_w40d4
```

The mass `m` is fixed by the TOML configuration and is not a neural-network
input. The solver uses one cell-centered native coordinate with `Nxx0`
samples plus explicit origin regularity. The TOML selects either the
first-order or second-order Robin condition, enforced at the endpoint or over
a physical-radius band. T001 uses the second-order condition for `r > 1e3`
and includes the outer endpoint.

### Fixed 2D

```bash
python -m pinndorama.solvers.punctures_2d.train \
  --config configs/punctures_2d/C002_w40d4.toml \
  --output-dir runs/C002_w40d4
```

### Parametric 2D

```bash
python -m pinndorama.solvers.punctures_2d_parametric.train \
  --config configs/punctures_2d_parametric/P001_w40d4.toml \
  --output-dir runs/P001_w40d4
```

The parameter `equal_spin_sz` is the raw physical Bowen--York component
`S_z`, with `S0_z = S1_z`. It is not the dimensionless spin `chi`.

### C04-to-C05 3D continuation

```bash
python -m pinndorama.solvers.punctures_3d.train \
  --config configs/punctures_3d/C04.toml \
  --output-dir runs/C04

python -m pinndorama.solvers.punctures_3d.train \
  --config configs/punctures_3d/C05.toml \
  --output-dir runs/C05 \
  --resume-checkpoint runs/C04/checkpoint_final.npz
```

C04 starts with Adam and continues with SSBroyden using uniform `xx0`
collocation. C05
requires the C04 checkpoint, disables Adam, changes to the configured two-zone
`xx0` sampling, and continues SSBroyden.

## Checkpoints and evaluation

Each trainer creates its requested output directory and writes the final model
to `checkpoint_final.npz`. Checkpoints contain float64 parameter arrays and
versioned JSON metadata. Resume validation rejects changes to the physical
problem, geometry, architecture, dtype, or ansatz, and records the parent
checkpoint hash. Pickle checkpoints are not supported.

Evaluate a checkpoint at explicit native-coordinate points with the shared
dispatcher. For example:

```bash
python -m pinndorama.reproducibility.evaluate punctures-3d \
  --config configs/punctures_3d/C04.toml \
  --checkpoint runs/C04/checkpoint_final.npz \
  --coords /path/to/native_xx_points.txt \
  --output /path/to/checkpoint_values.txt
```

The dispatcher accepts `toy-problem-1d`, `punctures-2d`,
`punctures-2d-parametric`, and `punctures-3d`. Parametric evaluation also requires
`--equal-spin-sz VALUE`. Inputs and outputs are always explicit; the tools do
not search for cluster files or bundled results.

## NRPyElliptic comparison

Build the standard reference solvers and the canonical 9-point reader with:

```bash
make -C reference_solvers/nrpyelliptic/axisymmetric/nrpyelliptic_conformally_flat
make -C reference_solvers/nrpyelliptic/three_dimensional/nrpyelliptic_conformally_flat
make -C reference_solvers/nrpyelliptic/three_dimensional/READER_nrpyelliptic_conformally_flat
```

NRPy solution binaries do not record every source parameter. Before comparing
a binary, create an authenticated reference TOML beside it. The comparison
workflow checks that metadata, evaluates the neural checkpoint, interpolates
the NRPy solution with the fixed 9-point reader, and reports a
volume-weighted relative L2 error. See the
[reproducibility guide](docs/reproducibility.md) for the complete commands and
file formats.

## Development and validation

Black is included in the normal installation. Install the test extra when
working on the code:

```bash
python -m pip install -e '.[test]'
```

Run the standard checks from the repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/pinndorama_pycache python -m py_compile \
  src/pinndorama/*.py src/pinndorama/solvers/*/*.py \
  src/pinndorama/reproducibility/*.py tests/*.py
python -m pytest
git diff --check
```

Tests that build C applications clean their generated files. Checkpoints,
solution binaries, executables, object files, plots, reports, and caches must
not be committed.

## Dependencies and provenance

The NRPy and SSBroyden source snapshots are stored in the repository so the
workflow does not depend on those upstream repositories remaining available.
Their exact commits and local changes are recorded in the corresponding
`VENDORED_FROM.md` files. The nested Optimistix code retains its Apache-2.0
license; written redistribution permission for the SSBroyden add-on files is
pending, and no broader project license is implied.
