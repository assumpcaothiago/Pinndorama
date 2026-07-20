"""Faithful full-residual, JIT-compiled Optimistix SSBroyden stage."""

from __future__ import annotations

import time
from typing import Any, Callable

import jax

from ..._ssbroyden import load_ssbroyden_api
from . import config, loss


def _copy_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: value.copy(), tree)


def run_ssbroyden_stage(
    params: Any,
    xx0: Any,
    xx1: Any,
    xx2: Any,
    axis_xx0: Any,
    axis_delta: Any,
    axis_phi: Any,
    periodic_xx0: Any,
    periodic_xx1: Any,
    outer_xx0: Any,
    outer_xx1: Any,
    outer_xx2: Any,
    *,
    source_terms: loss.SourceTerms,
    geometry: config.Geometry,
    physics: config.Physics,
    collocation: config.Collocation,
    max_steps: int,
    rtol: float,
    atol: float,
    log_every: int,
    progress_callback: Callable[[int, float, dict[str, float]], None] | None = None,
    checkpoint_callback: Callable[[int, Any, float], None] | None = None,
) -> tuple[Any, list[float], Any, dict[str, float], float, int, float, str]:
    """Optimize the complete loss and return its best accepted state."""

    SSBroyden, run_optimization = load_ssbroyden_api()
    started = time.time()
    history: list[float] = []

    def compute(current_params):
        return loss.compute_loss(
            current_params,
            xx0,
            xx1,
            xx2,
            axis_xx0=axis_xx0,
            axis_delta=axis_delta,
            axis_phi=axis_phi,
            periodic_xx0=periodic_xx0,
            periodic_xx1=periodic_xx1,
            outer_xx0=outer_xx0,
            outer_xx1=outer_xx1,
            outer_xx2=outer_xx2,
            source_terms=source_terms,
            geometry=geometry,
            physics=physics,
            collocation=collocation,
        )

    initial_total, _ = compute(params)
    best = {"params": _copy_tree(params), "loss": float(initial_total)}

    def objective(current_params):
        total, _ = compute(current_params)
        return total

    def callback(iteration: int, current_params: Any, _state: Any, current_loss: float):
        scalar_loss = float(current_loss)
        history.append(scalar_loss)
        if scalar_loss < best["loss"]:
            best["loss"] = scalar_loss
            best["params"] = _copy_tree(current_params)
        if progress_callback is not None and (
            iteration == 1 or iteration % log_every == 0
        ):
            total, components = compute(current_params)
            progress_callback(
                iteration,
                float(total),
                {name: float(value) for name, value in components.items()},
            )
        if checkpoint_callback is not None:
            checkpoint_callback(iteration, best["params"], float(best["loss"]))
        return False

    result = run_optimization(
        SSBroyden(rtol=rtol, atol=atol),
        objective,
        params,
        maxiter=max_steps,
        callback=callback,
        verbose=False,
    )
    best_params = best["params"]
    total, components = compute(best_params)
    return (
        best_params,
        history,
        total,
        {name: float(value) for name, value in components.items()},
        time.time() - started,
        int(result.num_iters),
        float(best["loss"]),
        str(result.result),
    )
