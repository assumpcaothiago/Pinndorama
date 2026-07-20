# NRPyElliptic 2D reference solver

`nrpyelliptic_conformally_flat/` is the axisymmetric NRPyElliptic reference
solver used for publication comparisons. It is generated C source and is kept
separate from the JAX implementation.

Build and inspect its command-line interface with:

```bash
make -C reference_solvers/nrpyelliptic/axisymmetric/nrpyelliptic_conformally_flat
reference_solvers/nrpyelliptic/axisymmetric/nrpyelliptic_conformally_flat/nrpyelliptic_conformally_flat --help
```

The executable accepts either its checked-in parameter file or an explicit
parameter-file path. Solution binaries do not contain every physical source
parameter, so publication evaluation additionally requires a reference TOML.
The obsolete 2D interpolation reader is preserved only in the dated archive;
all active comparisons use the canonical 3D 9-point reader.
