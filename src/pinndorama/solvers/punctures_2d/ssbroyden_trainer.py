"""Faithful full-JIT SSBroyden refinement for the publication 2D loss."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from ..._ssbroyden import load_ssbroyden_api
from . import config, loss


def _copy_tree(tree: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda value: value.copy(), tree)


def run_ssbroyden_stage(
    params: Any,
    xx0: Any,
    xx1: Any,
    parity_xx0: Any,
    parity_delta: Any,
    outer_xx0: Any,
    outer_xx1: Any,
    *,
    max_steps: int,
    rtol: float,
    atol: float,
    log_every: int,
    source_terms: loss.SourceTerms,
    geometry: Mapping[str, float],
    physics: Mapping[str, float],
    loss_weights: Mapping[str, float],
    progress_callback: Callable[[int, float, dict[str, float]], None] | None = None,
    checkpoint_callback: Callable[[int, Any, float, list[float]], None] | None = None,
    checkpoint_every: int = 0,
) -> tuple[Any, list[float], Any, dict[str, float], float, int, float, str]:
    """Run SSBroyden and return the best accepted-loss parameters."""

    config.enable_jax_x64()
    import jax

    SSBroyden, run_optimization = load_ssbroyden_api()

    geometry_kwargs = {
        "AMAX": float(geometry["AMAX"]),
        "bScale": float(geometry["bScale"]),
        "SINHWAA": float(geometry["SINHWAA"]),
    }
    loss_kwargs = {
        "physics": dict(physics),
        "interior_weight": float(loss_weights["interior_weight"]),
        "theta_parity_weight": float(loss_weights["theta_parity_weight"]),
        "outer_robin_weight": float(loss_weights["outer_robin_weight"]),
    }
    start_time = time.time()
    history: list[float] = []

    initial_total, _ = loss.compute_loss(
        params,
        xx0,
        xx1,
        parity_xx0=parity_xx0,
        parity_delta=parity_delta,
        outer_xx0=outer_xx0,
        outer_xx1=outer_xx1,
        source_terms=source_terms,
        **geometry_kwargs,
        **loss_kwargs,
    )
    best_state = {"params": _copy_tree(params), "loss": float(initial_total)}

    def objective_fn(current_params: Any):
        total, _ = loss.compute_loss(
            current_params,
            xx0,
            xx1,
            parity_xx0=parity_xx0,
            parity_delta=parity_delta,
            outer_xx0=outer_xx0,
            outer_xx1=outer_xx1,
            source_terms=source_terms,
            **geometry_kwargs,
            **loss_kwargs,
        )
        return total

    def callback(
        iteration: int, current_params: Any, _state: Any, current_loss: float
    ) -> bool:
        current_loss_float = float(current_loss)
        history.append(current_loss_float)
        if current_loss_float < best_state["loss"]:
            best_state["loss"] = current_loss_float
            best_state["params"] = _copy_tree(current_params)

        if progress_callback is not None and (
            iteration == 1 or iteration % log_every == 0 or iteration == max_steps
        ):
            total, components = loss.compute_loss(
                current_params,
                xx0,
                xx1,
                parity_xx0=parity_xx0,
                parity_delta=parity_delta,
                outer_xx0=outer_xx0,
                outer_xx1=outer_xx1,
                source_terms=source_terms,
                **geometry_kwargs,
                **loss_kwargs,
            )
            progress_callback(
                iteration,
                float(total),
                {name: float(value) for name, value in components.items()},
            )
        if (
            checkpoint_callback is not None
            and checkpoint_every > 0
            and iteration % checkpoint_every == 0
        ):
            checkpoint_callback(
                iteration,
                best_state["params"],
                float(best_state["loss"]),
                history,
            )
        return False

    result = run_optimization(
        SSBroyden(rtol=rtol, atol=atol),
        objective_fn,
        params,
        maxiter=max_steps,
        callback=callback,
        verbose=False,
    )

    best_params = best_state["params"]
    total, components = loss.compute_loss(
        best_params,
        xx0,
        xx1,
        parity_xx0=parity_xx0,
        parity_delta=parity_delta,
        outer_xx0=outer_xx0,
        outer_xx1=outer_xx1,
        source_terms=source_terms,
        **geometry_kwargs,
        **loss_kwargs,
    )
    return (
        best_params,
        history,
        total,
        {name: float(value) for name, value in components.items()},
        time.time() - start_time,
        int(result.num_iters),
        float(best_state["loss"]),
        str(result.result),
    )
