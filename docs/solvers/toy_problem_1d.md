# One-dimensional spherical toy solver

This solver is a compact manufactured test for the nonlinear elliptic training
workflow. The unknown is a spherically symmetric field $u(r)$, and the
positive mass $m$ is selected in the TOML configuration. It is not a neural
network input, so each checkpoint represents one fixed value of $m$.

## Equation and manufactured solution

The prescribed exact solution and background are

$$
u_{\mathrm{exact}}(r)=\frac{2m(1+r)}{r^2+2r+2},
\qquad
\psi(r)=1+\frac{m}{2r}.
$$

For a spherical scalar,

$$
\nabla^2u=u_{rr}+\frac{2}{r}u_r,
$$

and the nonlinear problem is

$$
u_{rr}+\frac{2}{r}u_r
+\frac18 A(r)[\psi(r)+u(r)]^{-7}=0.
$$

The source is constructed from the exact solution:

$$
A(r)=
\frac{m(r^2+6r+6)
[2r(r^2+2r+2)+m(5r^2+6r+2)]^7}
{4r^7(r^2+2r+2)^{10}}.
$$

Although $A$ and $\psi$ are separately singular at the origin, their
combination in the equation is finite. Training evaluates it in the equivalent
form

$$
N(r,u,m)=\frac18A(\psi+u)^{-7}
=\frac{4m(r^2+6r+6)
[2r(r^2+2r+2)+m(5r^2+6r+2)]^7}
{(r^2+2r+2)^{10}[m+2r(1+u)]^7}.
$$

No exact-solution values are supplied to the optimizer. The exact expression
is used only by tests and checkpoint evaluation.

## Native coordinates and loss

The network receives one native SinhSpherical coordinate $xx0$. The physical
radius is the NRPy map

$$
r(xx0)=\mathrm{AMPL}
\frac{\sinh(xx0/\mathrm{SINHW})}{\sinh(1/\mathrm{SINHW})},
\qquad 0\leq xx0\leq1.
$$

`Nxx0` counts uniform cell centers in this native coordinate; it is a PINN
collocation count and has no ghost zones. Analytic derivatives of the map
transform JAX derivatives from `xx0` to $r$.

The interior term is regularized by one power of radius:

$$
rR=r u_{rr}+2u_r+rN(r,u,m).
$$

The remaining loss terms impose

$$
u_r(0)=0,
$$

The outer loss has two independent configuration choices. The condition may
be the first-order Robin operator

$$
(1+r)(r u_r+u)=0,
$$

or the second-order Robin operator

$$
r^3\left(u_r+\frac{u}{r}\right)-2m=0.
$$

Set `outer_boundary_condition` to `first_order_robin` or
`second_order_robin`. Independently, set `outer_boundary_region` to
`endpoint` to evaluate the condition only at `xx0 = 1`, or to `radial_band`
to evaluate it at all cell-centered samples satisfying
`r > outer_boundary_r_min` and at the explicit endpoint. The physical cutoff
is required only for `radial_band` and must lie strictly between zero and
`AMPL`. Exactly one condition and one region are active in a run.

The checked configuration uses:

```toml
[loss]
outer_boundary_condition = "second_order_robin"
outer_boundary_region = "radial_band"
outer_boundary_r_min = 1.0e3
outer_boundary_weight = 1.0
```

The model uses the same smooth falloff as the puncture solvers,

$$
u=\frac{\mathrm{NN}(xx0)}{\sqrt{1+r^2}}.
$$

The second-order condition follows from

$$
u(r)=\frac{C}{r}-\frac{2m}{r^2}+\mathcal{O}(r^{-3}).
$$

It is asymptotic rather than an exact finite-radius identity, so its residual
is small but nonzero at the inner edge of the band. The first-order Robin
condition cancels the leading $C/r$ term but does not enforce the known
$-2m/r^2$ coefficient. Applying either operator over a band provides
information about the tail shape instead of constraining only one outer point.
The leading coefficient $C$ is never supplied to training.

## Training and continuation

```bash
python -m pinndorama.solvers.toy_problem_1d.train \
  --config configs/toy_problem_1d/T001_w40d4.toml \
  --output-dir runs/T001_w40d4
```

The allowed overrides are `--Nxx0`, `--seed`, `--adam-steps`, and
`--ssbroyden-steps`. Physics, geometry, architecture, loss definitions, and
optimizer tolerances remain in TOML. Adam is followed by the same pinned
SSBroyden runtime used by the puncture solvers.

Continuation uses a versioned NPZ checkpoint:

```bash
python -m pinndorama.solvers.toy_problem_1d.train \
  --config configs/toy_problem_1d/T001_w40d4.toml \
  --output-dir runs/T001_continued \
  --adam-steps 0 \
  --resume-checkpoint runs/T001_w40d4/checkpoint_final.npz
```

Changing `m`, the SinhSpherical geometry, architecture, dtype, or output
ansatz makes a checkpoint incompatible. Collocation and optimizer-stage
settings and the loss definition may change on continuation. The command above
uses the existing checkpoint as the initial model and applies only SSBroyden
refinement under the condition and region selected by the current TOML. For
example, changing `outer_boundary_condition` to `first_order_robin` and
`outer_boundary_region` to `endpoint` switches to the earlier endpoint loss;
in that case remove `outer_boundary_r_min` because endpoint mode does not use a
cutoff.

## Evaluation

Supply a one-column text file of native `xx0` values, or an NPZ containing
exactly an `xx0` array:

```bash
python -m pinndorama.reproducibility.evaluate toy-problem-1d \
  --config configs/toy_problem_1d/T001_w40d4.toml \
  --checkpoint runs/T001_w40d4/checkpoint_final.npz \
  --coords /path/to/xx0.txt \
  --output /path/to/toy_values.txt
```

The output columns are `xx0`, `r`, `u_nn`, `u_exact`, `absolute_error`, and
`relative_error`.

Plot the pointwise relative error against physical radius on logarithmic axes:

```bash
python -m pinndorama.solvers.toy_problem_1d.plot_error \
  --input /path/to/toy_values.txt \
  --output /path/to/toy_relative_error.png
```

The plotter omits samples with zero radius, zero relative error, or nonfinite
values because they cannot be represented on logarithmic axes. It does not
modify the evaluator's text output.
