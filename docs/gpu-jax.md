# Platform-specific GPU JAX installation

The root project metadata intentionally depends on the platform-neutral `jax`
package. CUDA/ROCm wheel selection depends on the host driver, accelerator,
and package index and must not be encoded in publication configurations.

Start with a supported Python 3.11 or 3.12 virtual environment and install the
repository's non-JAX dependencies. Then follow the current accelerator-specific
command in
the [official JAX installation documentation](https://docs.jax.dev/en/latest/installation.html)
for the machine being used. Only after that should the remaining project
dependencies be installed without allowing the package manager to replace the
selected JAX build.

Before a run, record:

```bash
python -c 'import jax; print(jax.__version__, jax.devices())'
python -m pip freeze
```

The solver enables `jax_enable_x64` and rejects a non-float64 checkpoint. A GPU
being visible is not sufficient: verify that float64 is supported and that the
reported device is the intended accelerator. CPU-only validation remains the
portable baseline used by the repository test suite.
