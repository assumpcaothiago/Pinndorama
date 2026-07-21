# GPU-enabled JAX installation

The default editable installation selects `jax[cuda12]` on Linux x86-64. This
installs matching pip-managed CUDA 12 runtime libraries, including cuDNN,
cuBLAS, cuSPARSE, and NCCL, together with the JAX CUDA plugin. The machine must
still provide a compatible NVIDIA driver, but a separately loaded CUDA toolkit
or cuDNN module is not required.

Create a supported Python 3.11 or 3.12 virtual environment and install the
repository normally:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Do not load system CUDA or cuDNN modules into the same runtime environment.
Their libraries can take precedence over the pip-managed copies and cause
version mismatches. In particular, a JAX plugin compiled for a newer cuDNN
minor release cannot run against an older module-provided cuDNN library.

On non-Linux-x86-64 platforms, the same installation command selects
platform-neutral JAX. Alternative accelerator stacks should follow the current
accelerator-specific instructions in the
[official JAX installation documentation](https://docs.jax.dev/en/latest/installation.html).

Validate the installation inside an allocated GPU compute job and record:

```bash
python -c 'import jax; print(jax.__version__, jax.devices())'
python -m pip freeze
```

The solver enables `jax_enable_x64` and rejects a non-float64 checkpoint. A GPU
being visible is not sufficient: execute at least one float64 operation and
verify that the reported device is the intended accelerator. CPU-only
validation remains the portable baseline used by the repository test suite.
