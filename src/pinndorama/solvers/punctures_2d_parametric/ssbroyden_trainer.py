"""Faithful full-residual SSBroyden refinement for publication training."""

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
    equal_spin_sz: Any,
    parity_xx0: Any,
    parity_delta: Any,
    parity_equal_spin_sz: Any,
    outer_xx0: Any,
    outer_xx1: Any,
    outer_equal_spin_sz: Any,
    *,
    max_steps: int | None = None,
    progress_callback: Callable[[int, float, dict[str, float]], None] | None = None,
    checkpoint_callback: Callable[[int, Any, float, list[float]], None] | None = None,
    checkpoint_step: int = 0,
    source_terms: loss.SourceTerms | None = None,
    geometry: dict[str, float] | None = None,
) -> tuple[Any, list[float], Any, dict[str, float], float, int, float, str]:
    """Run SSBroyden on the same complete loss used by Adam."""

    SSBroyden, run_optimization = load_ssbroyden_api()

    if max_steps is None:
        max_steps = config.ssbroyden_max_steps
    geometry_kwargs = {} if geometry is None else dict(geometry)

    ssb_start = time.time()
    ssb_loss_history: list[float] = []

    initial_total, initial_components = loss.compute_loss(
        params,
        xx0,
        xx1,
        equal_spin_sz,
        parity_xx0=parity_xx0,
        parity_delta=parity_delta,
        parity_equal_spin_sz=parity_equal_spin_sz,
        outer_xx0=outer_xx0,
        outer_xx1=outer_xx1,
        outer_equal_spin_sz=outer_equal_spin_sz,
        source_terms=source_terms,
        **geometry_kwargs,
    )
    best_state = {
        "params": _copy_tree(params),
        "loss": float(initial_total),
    }

    def objective_fn(current_params: Any):
        total, _components = loss.compute_loss(
            current_params,
            xx0,
            xx1,
            equal_spin_sz,
            parity_xx0=parity_xx0,
            parity_delta=parity_delta,
            parity_equal_spin_sz=parity_equal_spin_sz,
            outer_xx0=outer_xx0,
            outer_xx1=outer_xx1,
            outer_equal_spin_sz=outer_equal_spin_sz,
            source_terms=source_terms,
            **geometry_kwargs,
        )
        return total

    def callback(
        iter_idx: int, current_params: Any, _state: Any, current_loss: float
    ) -> bool:
        current_loss_float = float(current_loss)

        if current_loss_float < best_state["loss"]:
            best_state["loss"] = current_loss_float
            best_state["params"] = _copy_tree(current_params)

        ssb_loss_history.append(current_loss_float)

        should_report = progress_callback is not None and (
            iter_idx == 1 or iter_idx % config.ssbroyden_step == 0
        )
        should_checkpoint = (
            checkpoint_callback is not None
            and checkpoint_step > 0
            and iter_idx % checkpoint_step == 0
        )

        if should_report:
            total, components = loss.compute_loss(
                current_params,
                xx0,
                xx1,
                equal_spin_sz,
                parity_xx0=parity_xx0,
                parity_delta=parity_delta,
                parity_equal_spin_sz=parity_equal_spin_sz,
                outer_xx0=outer_xx0,
                outer_xx1=outer_xx1,
                outer_equal_spin_sz=outer_equal_spin_sz,
                source_terms=source_terms,
                **geometry_kwargs,
            )
            component_floats = {
                name: float(value) for name, value in components.items()
            }
            progress_callback(iter_idx, float(total), component_floats)

        if should_checkpoint:
            checkpoint_callback(
                iter_idx,
                best_state["params"],
                float(best_state["loss"]),
                ssb_loss_history,
            )
        return False

    solver = SSBroyden(
        rtol=config.ssbroyden_rtol,
        atol=config.ssbroyden_atol,
    )
    result = run_optimization(
        solver,
        objective_fn,
        params,
        maxiter=max_steps,
        callback=callback,
        verbose=False,
    )

    best_params = best_state["params"]
    total_loss, final_components = loss.compute_loss(
        best_params,
        xx0,
        xx1,
        equal_spin_sz,
        parity_xx0=parity_xx0,
        parity_delta=parity_delta,
        parity_equal_spin_sz=parity_equal_spin_sz,
        outer_xx0=outer_xx0,
        outer_xx1=outer_xx1,
        outer_equal_spin_sz=outer_equal_spin_sz,
        source_terms=source_terms,
        **geometry_kwargs,
    )
    final_component_floats = {
        name: float(value) for name, value in final_components.items()
    }
    elapsed = time.time() - ssb_start
    return (
        best_params,
        ssb_loss_history,
        total_loss,
        final_component_floats,
        elapsed,
        int(result.num_iters),
        float(best_state["loss"]),
        str(result.result),
    )
