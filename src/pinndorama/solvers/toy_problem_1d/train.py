"""Train the non-parametric 1D SinhSpherical toy PINN."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from . import checkpoint, config, coordinates, loss, model


def build_training_points(solver_config: config.SolverConfig, *, xp):
    collocation = solver_config.collocation
    return coordinates.cell_centered_xx0(
        Nxx0=collocation["Nxx0"],
        xx0_min=collocation["xx0_min"],
        xx0_max=collocation["xx0_max"],
        xp=xp,
    )


def _loss_kwargs(solver_config: config.SolverConfig) -> dict[str, Any]:
    return {
        "ampl": solver_config.geometry["AMPL"],
        "sinhw": solver_config.geometry["SINHW"],
        "m": solver_config.physics["m"],
        "interior_weight": solver_config.loss["interior_weight"],
        "origin_regularity_weight": solver_config.loss["origin_regularity_weight"],
        "outer_boundary_condition": solver_config.loss["outer_boundary_condition"],
        "outer_boundary_weight": solver_config.loss["outer_boundary_weight"],
    }


def _diagnostics(total: Any, components: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_loss": float(total),
        "components": {name: float(value) for name, value in components.items()},
    }


def _print_progress(
    stage: str, step: int, total: Any, components: dict[str, Any]
) -> None:
    values = {name: float(value) for name, value in components.items()}
    print(
        f"{stage} {step} | loss={float(total):.6e} | "
        f"rR={values['residual_loss']:.6e} | "
        f"origin={values['origin_regularity_loss']:.6e} | "
        f"outer_boundary={values['outer_boundary_loss']:.6e}",
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


def _make_adam_step(optimizer, solver_config: config.SolverConfig, *, jax, jnp, optax):
    kwargs = _loss_kwargs(solver_config)
    clip = jnp.asarray(solver_config.training["gradient_clip"], dtype=jnp.float64)

    @jax.jit
    def adam_step(params, optimizer_state, xx0, outer_xx0):
        (total, components), gradients = jax.value_and_grad(
            loss.compute_loss, has_aux=True
        )(params, xx0, outer_xx0, **kwargs)
        gradients = jax.tree_util.tree_map(
            lambda gradient: jnp.clip(gradient, -clip, clip), gradients
        )
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        return (
            optax.apply_updates(params, updates),
            optimizer_state,
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
        print(f"Continuing from {resume_checkpoint} (sha256={parent_hash})")
    else:
        params = model.init_mlp_params(
            jax.random.PRNGKey(solver_config.training["seed"]),
            solver_config.architecture["layers"],
            initialization=solver_config.architecture["initialization"],
        )

    xx0 = build_training_points(solver_config, xp=jnp)
    outer_xx0 = coordinates.outer_boundary_xx0(
        xx0,
        region=solver_config.loss["outer_boundary_region"],
        ampl=solver_config.geometry["AMPL"],
        sinhw=solver_config.geometry["SINHW"],
        r_min=solver_config.loss.get("outer_boundary_r_min"),
        xp=jnp,
    )
    print(
        f"Prepared {xx0.size} cell-centered xx0 samples and "
        f"{outer_xx0.size} outer boundary samples "
        f"({solver_config.loss['outer_boundary_condition']}, "
        f"{solver_config.loss['outer_boundary_region']})",
        flush=True,
    )
    kwargs = _loss_kwargs(solver_config)
    total, components = loss.compute_loss(params, xx0, outer_xx0, **kwargs)
    training = solver_config.training

    if training["adam_steps"] > 0:
        optimizer = _optimizer(training, optax=optax)
        optimizer_state = optimizer.init(params)
        adam_step = _make_adam_step(
            optimizer, solver_config, jax=jax, jnp=jnp, optax=optax
        )
        start = time.time()
        for step in range(1, training["adam_steps"] + 1):
            params, optimizer_state, total, components = adam_step(
                params, optimizer_state, xx0, outer_xx0
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
        print(f"Adam completed in {time.time() - start:.3f}s", flush=True)
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
                best_params, xx0, outer_xx0, **kwargs
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
            outer_xx0,
            max_steps=training["ssbroyden_steps"],
            rtol=training["ssbroyden_rtol"],
            atol=training["ssbroyden_atol"],
            log_every=training["log_every"],
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
    parser.add_argument("--config", required=True, help="Validated TOML configuration")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for NPZ checkpoints"
    )
    parser.add_argument("--resume-checkpoint", help="NPZ checkpoint to continue")
    parser.add_argument("--Nxx0", type=int, help="Override the xx0 collocation count")
    parser.add_argument("--seed", type=int, help="Override initialization seed")
    parser.add_argument("--adam-steps", type=int, help="Override Adam stage length")
    parser.add_argument(
        "--ssbroyden-steps", type=int, help="Override SSBroyden stage length"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        solver_config = config.load_config(args.config).with_overrides(
            Nxx0=args.Nxx0,
            seed=args.seed,
            adam_steps=args.adam_steps,
            ssbroyden_steps=args.ssbroyden_steps,
        )
    except config.ConfigError as error:
        parser.error(str(error))
    run_training(
        solver_config, args.output_dir, resume_checkpoint=args.resume_checkpoint
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
