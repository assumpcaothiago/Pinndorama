# NRPyElliptic comparison reader

This is the canonical publication-facing reader for both the 2D and 3D PINN
comparisons. It reads a three-dimensional NRPyElliptic solution binary and
interpolates the `uu` field with the validated tensor-product, nine-point
Lagrange stencil (nine source points in each native-coordinate direction).

Build it with:

```bash
make
```

Run it from any working directory. All paths are explicit:

```bash
nrpyell_reader --binary FILE --coords FILE --output FILE
```

`--binary` must use the versioned `NRPYELL3` binary format produced by the
active 3D NRPyElliptic solver. `--coords` is plain text containing one Cartesian
point per line, with exactly three whitespace-separated floating-point values:

```text
x y z
```

The output is a plain-text table with a header and four columns:

```text
# x y z uu
x y z interpolated_uu
```

Coordinates must lie within the interpolation domain, including the stencil
width. The binary carries five ghost cells; the comparison convention uses
four of them on either side of each target point to form the fixed nine-point
stencil.
