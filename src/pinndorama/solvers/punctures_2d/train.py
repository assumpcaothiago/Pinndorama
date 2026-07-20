"""Train the publication 2D SinhSymTP PINN from a checked-in TOML config."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from . import checkpoint, config, coordinates, loss, model


def build_training_points(solver_config: config.SolverConfig, *, xp):
    """Build cell-centered interior and structured boundary samples."""

    collocation = solver_config.collocation
    Nxx0 = collocation["Nxx0"]
    Nxx1 = collocation["Nxx1"]
    common = {
        "xx0_min": collocation["xx0_min"],
        "xx0_max": collocation["xx0_max"],
    }
    xx0, xx1 = coordinates.native_collocation_grid(
        Nxx0=Nxx0,
        Nxx1=Nxx1,
        xx1_min=collocation["xx1_min"],
        xx1_max=collocation["xx1_max"],
        xp=xp,
        **common,
    )
    parity_xx0, parity_delta = coordinates.theta_inner_parity_samples(
        Nxx0=Nxx0,
        Ndelta=Nxx1,
        delta_min=0.0,
        delta_max=collocation["xx1_parity_delta_max"],
        xp=xp,
        **common,
    )
    outer_xx0, outer_xx1 = coordinates.outer_robin_boundary_samples(
        Nxx1=Nxx1,
        xx0_value=collocation["xx0_max"],
        xx1_min=collocation["xx1_min"],
        xx1_max=collocation["xx1_max"],
        xp=xp,
    )
    return xx0, xx1, parity_xx0, parity_delta, outer_xx0, outer_xx1


def _loss_kwargs(solver_config: config.SolverConfig) -> dict[str, Any]:
    return {
        "AMAX": solver_config.geometry["AMAX"],
        "bScale": solver_config.geometry["bScale"],
        "SINHWAA": solver_config.geometry["SINHWAA"],
        "physics": solver_config.physics,
        "interior_weight": solver_config.loss["interior_weight"],
        "theta_parity_weight": solver_config.loss["theta_parity_weight"],
        "outer_robin_weight": solver_config.loss["outer_robin_weight"],
    }


def _components_as_floats(components: dict[str, Any]) -> dict[str, float]:
    return {name: float(value) for name, value in components.items()}


def _diagnostics(total: Any, components: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_loss": float(total),
        "components": _components_as_floats(components),
    }


def _print_progress(
    stage: str,
    step: int,
    total: Any,
    components: dict[str, Any],
) -> None:
    values = _components_as_floats(components)
    print(
        f"{stage} {step} | loss={float(total):.6e} | "
        f"JR_H={values['residual_loss']:.6e} | "
        f"theta_parity={values['theta_inner_parity_loss']:.6e} | "
        f"outer_robin={values['outer_robin_boundary_loss']:.6e}",
        flush=True,
    )


def _optimizer(training: dict[str, Any], *, optax):
    decay_steps = max(int(training["adam_steps"]), 1)
    schedule = optax.cosine_decay_schedule(
        init_value=training["adam_learning_rate"],
        decay_steps=decay_steps,
        alpha=training["adam_min_learning_rate"] / training["adam_learning_rate"],
    )
    return optax.adam(schedule)


def _make_adam_step(
    optimizer,
    solver_config: config.SolverConfig,
    *,
    jax,
    jnp,
    optax,
):
    kwargs = _loss_kwargs(solver_config)
    clip = jnp.asarray(solver_config.training["gradient_clip"], dtype=jnp.float64)

    @jax.jit
    def adam_step(
        params,
        optimizer_state,
        xx0,
        xx1,
        parity_xx0,
        parity_delta,
        outer_xx0,
        outer_xx1,
        psi_background,
        add_times_auu,
    ):
        (total, components), gradients = jax.value_and_grad(
            loss.compute_loss,
            has_aux=True,
        )(
            params,
            xx0,
            xx1,
            parity_xx0=parity_xx0,
            parity_delta=parity_delta,
            outer_xx0=outer_xx0,
            outer_xx1=outer_xx1,
            source_terms=(psi_background, add_times_auu),
            **kwargs,
        )
        gradients = jax.tree_util.tree_map(
            lambda gradient: jnp.clip(gradient, -clip, clip), gradients
        )
        updates, new_optimizer_state = optimizer.update(
            gradients, optimizer_state, params
        )
        return (
            optax.apply_updates(params, updates),
            new_optimizer_state,
            total,
            components,
        )

    return adam_step


def run_training(
    solver_config: config.SolverConfig,
    output_dir: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    """Execute Adam warm-up and faithful SSBroyden refinement."""

    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp
    import optax

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    parent_hash: str | None = None
    if resume_checkpoint is not None:
        loaded_params, loaded_metadata = checkpoint.load_checkpoint(resume_checkpoint)
        checkpoint.validate_resume(loaded_metadata, solver_config)
        params = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value, dtype=jnp.float64), loaded_params
        )
        parent_hash = checkpoint.sha256_file(resume_checkpoint)
        print(
            f"Continuing from {resume_checkpoint} (sha256={parent_hash})",
            flush=True,
        )
    else:
        key = jax.random.PRNGKey(solver_config.training["seed"])
        params = model.init_mlp_params(
            key,
            solver_config.architecture["layers"],
            initialization=solver_config.architecture["initialization"],
        )

    points = build_training_points(solver_config, xp=jnp)
    xx0, xx1, parity_xx0, parity_delta, outer_xx0, outer_xx1 = points
    source_start = time.time()
    source_terms = loss.precompute_source_terms(
        xx0,
        xx1,
        AMAX=solver_config.geometry["AMAX"],
        bScale=solver_config.geometry["bScale"],
        SINHWAA=solver_config.geometry["SINHWAA"],
        physics=solver_config.physics,
    )
    source_terms = jax.tree_util.tree_map(
        lambda value: value.block_until_ready(), source_terms
    )
    print(
        f"Prepared {xx0.size} cell-centered interior samples in "
        f"{time.time() - source_start:.3f}s",
        flush=True,
    )

    common_loss_args = {
        "parity_xx0": parity_xx0,
        "parity_delta": parity_delta,
        "outer_xx0": outer_xx0,
        "outer_xx1": outer_xx1,
        "source_terms": source_terms,
        **_loss_kwargs(solver_config),
    }
    total, components = loss.compute_loss(params, xx0, xx1, **common_loss_args)
    training = solver_config.training

    if training["adam_steps"] > 0:
        optimizer = _optimizer(training, optax=optax)
        optimizer_state = optimizer.init(params)
        adam_step = _make_adam_step(
            optimizer,
            solver_config,
            jax=jax,
            jnp=jnp,
            optax=optax,
        )
        for step in range(1, training["adam_steps"] + 1):
            params, optimizer_state, total, components = adam_step(
                params,
                optimizer_state,
                xx0,
                xx1,
                parity_xx0,
                parity_delta,
                outer_xx0,
                outer_xx1,
                source_terms[0],
                source_terms[1],
            )
            if (
                step == 1
                or step % training["log_every"] == 0
                or step == training["adam_steps"]
            ):
                _print_progress("Adam", step, total, components)
            if step % training["checkpoint_every"] == 0:
                metadata = checkpoint.build_metadata(
                    solver_config,
                    stage="adam",
                    step=step,
                    parent_checkpoint_sha256=parent_hash,
                    diagnostics=_diagnostics(total, components),
                )
                checkpoint.save_checkpoint(
                    destination / f"checkpoint_adam_{step:08d}.npz",
                    params,
                    metadata,
                )

        metadata = checkpoint.build_metadata(
            solver_config,
            stage="adam",
            step=training["adam_steps"],
            parent_checkpoint_sha256=parent_hash,
            diagnostics=_diagnostics(total, components),
        )
        checkpoint.save_checkpoint(
            destination / "checkpoint_adam_final.npz", params, metadata
        )

    ssb_iterations = 0
    ssb_result = "not_run"
    if training["ssbroyden_steps"] > 0:
        from .ssbroyden_trainer import run_ssbroyden_stage

        def progress(
            iteration: int,
            current_total: float,
            current_components: dict[str, float],
        ) -> None:
            _print_progress("SSBroyden", iteration, current_total, current_components)

        def save_ssb(
            iteration: int,
            best_params: Any,
            best_loss: float,
            _history: list[float],
        ) -> None:
            checkpoint_total, checkpoint_components = loss.compute_loss(
                best_params, xx0, xx1, **common_loss_args
            )
            metadata = checkpoint.build_metadata(
                solver_config,
                stage="ssbroyden",
                step=iteration,
                parent_checkpoint_sha256=parent_hash,
                diagnostics={
                    **_diagnostics(checkpoint_total, checkpoint_components),
                    "best_accepted_loss": best_loss,
                },
            )
            checkpoint.save_checkpoint(
                destination / f"checkpoint_ssbroyden_{iteration:08d}.npz",
                best_params,
                metadata,
            )

        (
            params,
            _history,
            total,
            component_values,
            elapsed,
            ssb_iterations,
            best_loss,
            ssb_result,
        ) = run_ssbroyden_stage(
            params,
            xx0,
            xx1,
            parity_xx0,
            parity_delta,
            outer_xx0,
            outer_xx1,
            max_steps=training["ssbroyden_steps"],
            rtol=training["ssbroyden_rtol"],
            atol=training["ssbroyden_atol"],
            log_every=training["log_every"],
            source_terms=source_terms,
            geometry=solver_config.geometry,
            physics=solver_config.physics,
            loss_weights=solver_config.loss,
            progress_callback=progress,
            checkpoint_callback=save_ssb,
            checkpoint_every=training["checkpoint_every"],
        )
        components = {
            name: jnp.asarray(value) for name, value in component_values.items()
        }
        print(
            f"SSBroyden completed {ssb_iterations} accepted steps in {elapsed:.3f}s; "
            f"best loss={best_loss:.6e}; result={ssb_result}",
            flush=True,
        )

    final_metadata = checkpoint.build_metadata(
        solver_config,
        stage="complete",
        step=(
            ssb_iterations
            if training["ssbroyden_steps"] > 0
            else training["adam_steps"]
        ),
        parent_checkpoint_sha256=parent_hash,
        diagnostics={**_diagnostics(total, components), "ssbroyden_result": ssb_result},
    )
    final_path = destination / "checkpoint_final.npz"
    checkpoint.save_checkpoint(final_path, params, final_metadata)
    print(f"Saved {final_path}", flush=True)
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Publication TOML configuration"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for NPZ checkpoints"
    )
    parser.add_argument(
        "--resume-checkpoint", help="Publication NPZ checkpoint to continue"
    )
    parser.add_argument("--Nxx0", type=int, help="Override the xx0 collocation count")
    parser.add_argument("--Nxx1", type=int, help="Override the xx1 collocation count")
    parser.add_argument("--seed", type=int, help="Override initialization seed")
    parser.add_argument("--adam-steps", type=int, help="Override Adam stage length")
    parser.add_argument(
        "--ssbroyden-steps", type=int, help="Override SSBroyden stage length"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        solver_config = config.load_config(args.config).with_overrides(
            Nxx0=args.Nxx0,
            Nxx1=args.Nxx1,
            seed=args.seed,
            adam_steps=args.adam_steps,
            ssbroyden_steps=args.ssbroyden_steps,
        )
    except config.ConfigError as error:
        parser.error(str(error))
    run_training(
        solver_config,
        args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
