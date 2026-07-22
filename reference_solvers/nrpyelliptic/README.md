# NRPyElliptic reference applications

This directory contains the publication-facing C tools:

- `nrpyelliptic_conformally_flat_3D/`: fully 3D GW150914-like reference;
- `nrpyelliptic_conformally_flat_symmetricID/`: axisymmetric physical test on
  a full 3D grid, with the `bScale=2.5` coordinate map used by the 2D PINN;
- `READER_nrpyelliptic_conformally_flat/`: canonical fixed 9-point reader;
- `nrpyelliptic_conformally_flat_symmetricID-NN_guess/`: equal-spin
  parametric-neural-network warm-start variant.

Each solver directory builds independently with `make` and contains exactly
one `nrpyelliptic_conformally_flat.par` matched to its physical source and
compiled coordinate map. Run the executable without a parfile argument from
its own directory to select that local file. The reader has no repository- or
cluster-specific defaults. The warm-start solver requires a generated header
with complete checkpoint and physics provenance before its neural-network mode
can be built.

Both reference examples are structurally three-dimensional. The `symmetricID`
name refers to the axisymmetry of its physical source and solution, not to a
two-dimensional numerical grid or binary format.

NRPy solution binaries do not encode all physical parameters. Evaluation code
therefore requires an accompanying reference TOML and reports the
volume-weighted relative L2 norm using the canonical reader.
