"""Train the publication 3D native SinhSymTP PINN from a validated TOML file."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from . import checkpoint, config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Validated publication TOML")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for NPZ checkpoints"
    )
    parser.add_argument(
        "--resume-checkpoint", help="Portable NPZ checkpoint to continue"
    )
    parser.add_argument("--Nxx0", type=int)
    parser.add_argument("--Nxx1", type=int)
    parser.add_argument("--Nxx2", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--ssbroyden-steps", type=int)
    return parser.parse_args(argv)


def _load_run(args: argparse.Namespace) -> config.RunConfig:
    run = config.load(args.config)
    run = config.with_overrides(
        run,
        Nxx0=args.Nxx0,
        Nxx1=args.Nxx1,
        Nxx2=args.Nxx2,
        seed=args.seed,
        adam_steps=args.adam_steps,
        ssbroyden_steps=args.ssbroyden_steps,
    )
    if run.continuation.require_resume and args.resume_checkpoint is None:
        raise ValueError(f"configuration {run.name!r} requires --resume-checkpoint")
    return run


def _build_training_points(run: config.RunConfig):
    import jax.numpy as jnp
    from . import coordinates

    residual = coordinates.native_collocation_grid(run.collocation, xp=jnp)
    axis = coordinates.theta_axis_regularization_samples(run.collocation, xp=jnp)
    periodic = coordinates.phi_periodicity_samples(run.collocation, xp=jnp)
    outer = coordinates.outer_robin_boundary_samples(run.collocation, xp=jnp)
    return (*residual, *axis, *periodic, *outer)


def _compute_loss(params, points, source_terms, run: config.RunConfig):
    from . import loss

    (
        xx0,
        xx1,
        xx2,
        axis_xx0,
        axis_delta,
        axis_phi,
        periodic_xx0,
        periodic_xx1,
        outer_xx0,
        outer_xx1,
        outer_xx2,
    ) = points
    return loss.compute_loss(
        params,
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
        geometry=run.geometry,
        physics=run.physics,
        collocation=run.collocation,
    )


def _metrics(total, components: dict[str, Any]) -> dict[str, float]:
    return {
        "total_loss": float(total),
        **{name: float(value) for name, value in components.items()},
    }


def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} | loss={metrics['total_loss']:.6e} | "
        f"residual={metrics['residual_loss']:.6e} | "
        f"theta_axis={metrics['theta_axis_regularity_loss']:.6e} | "
        f"phi_periodicity={metrics['phi_periodicity_loss']:.6e} | "
        f"outer_robin={metrics['outer_robin_boundary_loss']:.6e}",
        flush=True,
    )


def _save(
    output_dir: Path,
    filename: str,
    params,
    run: config.RunConfig,
    *,
    stage: str,
    step: int,
    parent_sha256: str | None,
    metrics: dict[str, float],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    import jax
    from . import model

    metadata = checkpoint.build_metadata(
        run,
        stage=stage,
        step=step,
        parent_sha256=parent_sha256,
        metrics=metrics,
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    path = output_dir / filename
    leaves = [jax.device_get(value) for value in model.parameter_leaves(params)]
    digest = checkpoint.save(path, leaves, metadata)
    print(f"checkpoint={path} sha256={digest}", flush=True)
    return path


def _make_adam_step(run: config.RunConfig):
    import jax
    import jax.numpy as jnp
    import optax

    decay_steps = max(run.training.adam_steps, 1)
    schedule = optax.cosine_decay_schedule(
        init_value=run.training.learning_rate,
        decay_steps=decay_steps,
        alpha=run.training.min_learning_rate / run.training.learning_rate,
    )
    optimizer = optax.adam(schedule)
    clip_value = jnp.asarray(run.training.gradient_clip_value, dtype=jnp.float64)

    def objective(params, points, source_terms):
        return _compute_loss(params, points, source_terms, run)

    @jax.jit
    def step(params, optimizer_state, points, source_terms):
        (total, components), gradients = jax.value_and_grad(
            objective,
            has_aux=True,
        )(params, points, source_terms)
        gradients = jax.tree_util.tree_map(
            lambda gradient: jnp.clip(gradient, -clip_value, clip_value),
            gradients,
        )
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        return (
            optax.apply_updates(params, updates),
            optimizer_state,
            total,
            components,
        )

    return optimizer, step


def run_training(
    run: config.RunConfig,
    *,
    output_dir: str | Path,
    resume_checkpoint: str | Path | None,
) -> Path:
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp
    from . import loss, model

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    parent_sha256: str | None = None
    if resume_checkpoint is None:
        key = jax.random.PRNGKey(run.training.seed)
        params = model.init_mlp_params(key, run.architecture.layers)
        origin = "fresh initialization"
    else:
        leaves, parent_metadata, parent_sha256 = checkpoint.load(resume_checkpoint)
        checkpoint.validate_resume(run, parent_metadata)
        params = model.params_from_leaves(leaves, run.architecture.layers)
        origin = f"resume {Path(resume_checkpoint).expanduser().resolve()}"

    print(
        f"run={run.name} origin={origin} dtype={run.dtype} "
        f"layers={list(run.architecture.layers)}",
        flush=True,
    )
    print(
        f"collocation={run.collocation.Nxx0}x"
        f"{run.collocation.Nxx1}x{run.collocation.Nxx2} "
        f"xx0_sampling={run.collocation.xx0_sampling}",
        flush=True,
    )

    points = _build_training_points(run)
    xx0, xx1, xx2 = points[:3]
    source_started = time.time()
    source_terms = loss.precompute_source_terms(
        xx0,
        xx1,
        xx2,
        run.geometry,
        run.physics,
    )
    source_terms = jax.tree_util.tree_map(
        lambda value: value.block_until_ready(), source_terms
    )
    print(f"source_precompute_seconds={time.time() - source_started:.3f}", flush=True)

    total, components = _compute_loss(params, points, source_terms, run)
    current_metrics = _metrics(total, components)
    _print_metrics("initial", current_metrics)
    last_stage = "initialized"
    last_step = 0

    if run.training.adam_steps > 0:
        optimizer, adam_step = _make_adam_step(run)
        optimizer_state = optimizer.init(params)
        for step_index in range(1, run.training.adam_steps + 1):
            params, optimizer_state, total, components = adam_step(
                params,
                optimizer_state,
                points,
                source_terms,
            )
            if (
                step_index == 1
                or step_index % run.training.log_every == 0
                or step_index == run.training.adam_steps
            ):
                current_metrics = _metrics(total, components)
                _print_metrics(
                    f"adam {step_index}/{run.training.adam_steps}", current_metrics
                )
            if (
                step_index % run.training.checkpoint_every == 0
                and step_index < run.training.adam_steps
            ):
                current_metrics = _metrics(total, components)
                _save(
                    output_path,
                    f"adam_step_{step_index:08d}.npz",
                    params,
                    run,
                    stage="adam",
                    step=step_index,
                    parent_sha256=parent_sha256,
                    metrics=current_metrics,
                )
        total, components = _compute_loss(params, points, source_terms, run)
        current_metrics = _metrics(total, components)
        _save(
            output_path,
            f"adam_step_{run.training.adam_steps:08d}.npz",
            params,
            run,
            stage="adam",
            step=run.training.adam_steps,
            parent_sha256=parent_sha256,
            metrics=current_metrics,
        )
        last_stage = "adam"
        last_step = run.training.adam_steps

    ssb_result = "not_run"
    if run.training.ssbroyden_steps > 0:
        from .ssbroyden_trainer import run_ssbroyden_stage

        (
            xx0,
            xx1,
            xx2,
            axis_xx0,
            axis_delta,
            axis_phi,
            periodic_xx0,
            periodic_xx1,
            outer_xx0,
            outer_xx1,
            outer_xx2,
        ) = points

        def progress(iteration: int, _total: float, component_values: dict[str, float]):
            _print_metrics(
                f"ssbroyden {iteration}/{run.training.ssbroyden_steps}",
                {"total_loss": _total, **component_values},
            )

        def save_best(iteration: int, best_params, best_loss: float):
            if (
                iteration % run.training.checkpoint_every != 0
                or iteration >= run.training.ssbroyden_steps
            ):
                return
            _save(
                output_path,
                f"ssbroyden_step_{iteration:08d}.npz",
                best_params,
                run,
                stage="ssbroyden",
                step=iteration,
                parent_sha256=parent_sha256,
                metrics={"total_loss": best_loss},
            )

        (
            params,
            _ssb_history,
            total,
            component_values,
            elapsed,
            accepted_steps,
            best_loss,
            ssb_result,
        ) = run_ssbroyden_stage(
            params,
            xx0,
            xx1,
            xx2,
            axis_xx0,
            axis_delta,
            axis_phi,
            periodic_xx0,
            periodic_xx1,
            outer_xx0,
            outer_xx1,
            outer_xx2,
            source_terms=source_terms,
            geometry=run.geometry,
            physics=run.physics,
            collocation=run.collocation,
            max_steps=run.training.ssbroyden_steps,
            rtol=run.training.ssbroyden_rtol,
            atol=run.training.ssbroyden_atol,
            log_every=run.training.log_every,
            progress_callback=progress,
            checkpoint_callback=save_best,
        )
        current_metrics = {"total_loss": float(total), **component_values}
        _print_metrics(
            f"ssbroyden complete accepted={accepted_steps} seconds={elapsed:.3f}",
            current_metrics,
        )
        last_stage = "ssbroyden"
        last_step = accepted_steps

    final_path = _save(
        output_path,
        "checkpoint_final.npz",
        params,
        run,
        stage=last_stage,
        step=last_step,
        parent_sha256=parent_sha256,
        metrics=current_metrics,
        extra_metadata={"optimizer_result": ssb_result},
    )
    return final_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run = _load_run(args)
        run_training(
            run,
            output_dir=args.output_dir,
            resume_checkpoint=args.resume_checkpoint,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
