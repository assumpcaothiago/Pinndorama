# NRPyElliptic 3D reference tools

This directory contains the publication-facing C tools:

- `nrpyelliptic_conformally_flat/`: standard 3D reference solver;
- `READER_nrpyelliptic_conformally_flat/`: canonical fixed 9-point reader;
- `nrpyelliptic_conformally_flat_symmetricID-NN_guess/`: equal-spin
  parametric-neural-network warm-start variant.

Each subdirectory builds independently with `make`. The standard solver accepts
an explicit parameter-file path. The reader has no repository- or
cluster-specific defaults. The warm-start solver requires a generated header
with complete checkpoint and physics provenance before its neural-network mode
can be built.

NRPy solution binaries do not encode all physical parameters. Evaluation code
therefore requires an accompanying reference TOML and reports the
volume-weighted relative L2 norm using the canonical reader.
