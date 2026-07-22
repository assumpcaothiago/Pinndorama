# Parametric-NN warm start

This generated NRPyElliptic variant supports two runtime initializers:

```text
initial_guess = "zero"
initial_guess = "parametric_nn"
```

Restart files take precedence: when the normal NRPyElliptic checkpoint is
present, it is loaded before the initializer is inspected.

## Generate the required header

No neural-network weights are embedded in the repository. Build the header
from a user-supplied publication checkpoint, then compile:

```bash
make nn-header CHECKPOINT=/path/to/parametric_checkpoint.npz
make
```

The checkpoint must contain exactly ten float64 model arrays named
`leaf_000` through `leaf_009`, alternating dense weights and biases in pytree
order. Every weight must use Equinox Linear's `(out_features, in_features)`
layout. The fixed architecture is `[3, 40, 40, 40, 40, 1]` (`w40d4`).

Metadata must be embedded under `metadata_json` as a one-dimensional `uint8`
array containing UTF-8 JSON. Sidecars, string arrays, multidimensional byte
arrays, aliases, and historical spellings are rejected. The canonical fields
and P001 values are:

```json
{
  "schema_version": 1,
  "dtype": "float64",
  "architecture": {"layers": [3, 40, 40, 40, 40, 1], "activation": "tanh"},
  "geometry": {
    "coordinate_system": "SinhSymTP",
    "AMAX": 1000000.0, "bScale": 2.5, "SINHWAA": 0.07
  },
  "physics": {
    "bare_mass_0": 0.5, "bare_mass_1": 0.5, "zPunc": 2.5,
    "P0_x": 0.0, "P0_y": 0.0, "P0_z": 0.0,
    "P1_x": 0.0, "P1_y": 0.0, "P1_z": 0.0,
    "S0_x": 0.0, "S0_y": 0.0, "S1_x": 0.0, "S1_y": 0.0
  },
  "parameter": {
    "equal_spin_sz": {"minimum": -0.2, "maximum": 0.2, "sampling": "cell_centered"}
  },
  "ansatz": {"output_transform": "smooth_inverse_radius"}
}
```

Current unrelated top-level fields such as collocation, training, stage, step,
and parent provenance are allowed. The generated header records SHA-256 hashes
of both the NPZ checkpoint and its embedded metadata.

At runtime `parametric_nn` validates the SinhSymTP geometry, every fixed
Bowen-York source parameter, equal raw spins `S0_z = S1_z`, and the exported
`equal_spin_sz` range once before the OpenMP initialization loop. The spin
input is raw Bowen-York `S_z` throughout. See the paired
`nrpyelliptic_conformally_flat_zero.par` and
`nrpyelliptic_conformally_flat_equal_spin_nn.par` examples.
