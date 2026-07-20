"""Train the publication parametric 2D SinhSymTP PINN."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Mapping

from . import checkpoint, config


def cell_centered_values(minimum: float, maximum: float, points: int, *, xp):
    """Return cell centers spanning ``[minimum, maximum]``."""

    if points <= 0:
        raise ValueError("points must be positive")
    if not minimum < maximum:
        raise ValueError("minimum must be smaller than maximum")
    spacing = (maximum - minimum) / points
    return minimum + (xp.arange(points, dtype=xp.float64) + 0.5) * spacing


def build_training_points(cfg: Mapping[str, Any], *, xp):
    """Build residual, theta-parity, and Robin samples for every spin cell."""

    from . import coordinates

    collocation = cfg["collocation"]
    domain = cfg["domain"]
    parameter = cfg["parameter"]["equal_spin_sz"]
    Nxx0 = collocation["Nxx0"]
    Nxx1 = collocation["Nxx1"]
    spin_points = collocation["equal_spin_sz_points"]
    spin_cells = cell_centered_values(
        parameter["minimum"], parameter["maximum"], spin_points, xp=xp
    )

    xx0_base, xx1_base = coordinates.native_collocation_grid(
        Nxx0=Nxx0,
        Nxx1=Nxx1,
        xx0_min=domain["xx0_min"],
        xx0_max=domain["xx0_max"],
        xx1_min=domain["xx1_min"],
        xx1_max=domain["xx1_max"],
        xp=xp,
    )
    xx0 = xp.repeat(xx0_base, spin_points)
    xx1 = xp.repeat(xx1_base, spin_points)
    residual_equal_spin_sz = xp.tile(spin_cells, xx0_base.size)

    parity_xx0_base, parity_delta_base = coordinates.theta_inner_parity_samples(
        Nxx0=Nxx0,
        Ndelta=Nxx1,
        xx0_min=domain["xx0_min"],
        xx0_max=domain["xx0_max"],
        delta_min=domain["xx1_min"],
        delta_max=domain["xx1_parity_delta_max"],
        xp=xp,
    )
    parity_xx0 = xp.repeat(parity_xx0_base, spin_points)
    parity_delta = xp.repeat(parity_delta_base, spin_points)
    parity_equal_spin_sz = xp.tile(spin_cells, parity_xx0_base.size)

    outer_xx0_base, outer_xx1_base = coordinates.outer_robin_boundary_samples(
        Nxx1=Nxx1,
        xx0_value=domain["xx0_max"],
        xx1_min=domain["xx1_min"],
        xx1_max=domain["xx1_max"],
        xp=xp,
    )
    outer_xx0 = xp.repeat(outer_xx0_base, spin_points)
    outer_xx1 = xp.repeat(outer_xx1_base, spin_points)
    outer_equal_spin_sz = xp.tile(spin_cells, outer_xx0_base.size)
    return (
        xx0,
        xx1,
        residual_equal_spin_sz,
        parity_xx0,
        parity_delta,
        parity_equal_spin_sz,
        outer_xx0,
        outer_xx1,
        outer_equal_spin_sz,
    )


def _create_optimizer(optax, adam_cfg: Mapping[str, Any]):
    decay_steps = max(int(adam_cfg["steps"]), 1)
    alpha = adam_cfg["min_learning_rate"] / adam_cfg["learning_rate"]
    schedule = optax.cosine_decay_schedule(
        init_value=adam_cfg["learning_rate"], decay_steps=decay_steps, alpha=alpha
    )
    return optax.adam(schedule)


def make_train_step(loss_module, optimizer, *, gradient_clip: float, geometry):
    import jax
    import jax.numpy as jnp
    import optax

    @jax.jit
    def train_step(
        params,
        optimizer_state,
        xx0,
        xx1,
        equal_spin_sz,
        parity_xx0,
        parity_delta,
        parity_equal_spin_sz,
        outer_xx0,
        outer_xx1,
        outer_equal_spin_sz,
        source_psi_background,
        source_add_times_auu,
    ):
        (total, components), gradients = jax.value_and_grad(
            loss_module.compute_loss, has_aux=True
        )(
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
            source_terms=(source_psi_background, source_add_times_auu),
            **geometry,
        )
        clip = jnp.asarray(gradient_clip, dtype=jnp.float64)
        gradients = jax.tree_util.tree_map(
            lambda value: jnp.clip(value, -clip, clip), gradients
        )
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, total, components

    return train_step


def _component_floats(components: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(value) for name, value in components.items()}


def _save(
    path: Path,
    params: Any,
    cfg: Mapping[str, Any],
    *,
    stage: str,
    step: int,
    parent_hash: str | None,
    total: Any,
    components: Mapping[str, Any],
) -> None:
    metadata = checkpoint.build_metadata(
        cfg,
        stage=stage,
        step=step,
        parent_checkpoint_sha256=parent_hash,
        loss_value=float(total),
        loss_components=_component_floats(components),
    )
    checkpoint.save_checkpoint(path, params, metadata)
    print(f"Saved {stage} checkpoint: {path}", flush=True)


def run_training(
    cfg: Mapping[str, Any],
    *,
    output_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    """Run Adam warm-up followed by faithful SSBroyden refinement."""

    cfg = config.validate_config(cfg)
    config.apply_runtime_config(cfg)
    config.enable_jax_x64()

    import jax
    import jax.numpy as jnp
    import optax
    from . import loss, model

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parent_hash: str | None = None
    resume_metadata: dict[str, Any] | None = None
    if resume_checkpoint is None:
        key = jax.random.PRNGKey(cfg["seed"])
        params = model.init_mlp_params(key, layers=cfg["architecture"]["layers"])
    else:
        resume_path = Path(resume_checkpoint)
        loaded_params, resume_metadata = checkpoint.load_checkpoint(resume_path)
        checkpoint.validate_immutable_metadata(cfg, resume_metadata)
        params = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value, dtype=jnp.float64), loaded_params
        )
        parent_hash = checkpoint.file_sha256(resume_path)
        print(
            f"Resuming {resume_path} (sha256={parent_hash}, "
            f"stage={resume_metadata['stage']}, step={resume_metadata['step']})",
            flush=True,
        )

    geometry = config.geometry_parameter_dict(cfg)
    points = build_training_points(cfg, xp=jnp)
    (
        xx0,
        xx1,
        residual_equal_spin_sz,
        parity_xx0,
        parity_delta,
        parity_equal_spin_sz,
        outer_xx0,
        outer_xx1,
        outer_equal_spin_sz,
    ) = points
    spin = cfg["parameter"]["equal_spin_sz"]
    print(
        "Collocation: "
        f"{cfg['collocation']['Nxx0']} x "
        f"{cfg['collocation']['Nxx1']} x "
        f"{cfg['collocation']['equal_spin_sz_points']}; "
        f"equal_spin_sz in [{spin['minimum']}, {spin['maximum']}] "
        "(raw Bowen-York S_z, cell-centered, S0_z=S1_z)",
        flush=True,
    )
    source_start = time.time()
    source_terms = loss.precompute_source_terms(
        xx0,
        xx1,
        residual_equal_spin_sz,
        physics=cfg["physics"],
        **geometry,
    )
    source_terms = jax.tree_util.tree_map(
        lambda value: value.block_until_ready(), source_terms
    )
    print(f"Source precomputation: {time.time() - source_start:.3f}s", flush=True)

    loss_arguments = dict(
        parity_xx0=parity_xx0,
        parity_delta=parity_delta,
        parity_equal_spin_sz=parity_equal_spin_sz,
        outer_xx0=outer_xx0,
        outer_xx1=outer_xx1,
        outer_equal_spin_sz=outer_equal_spin_sz,
        source_terms=source_terms,
        **geometry,
    )
    total, components = loss.compute_loss(
        params, xx0, xx1, residual_equal_spin_sz, **loss_arguments
    )

    adam_cfg = cfg["training"]["adam"]
    adam_start_step = 0
    if resume_metadata is not None and resume_metadata["stage"] == "adam":
        adam_start_step = int(resume_metadata["step"])
    if adam_cfg["steps"] > 0:
        optimizer = _create_optimizer(optax, adam_cfg)
        optimizer_state = optimizer.init(params)
        train_step = make_train_step(
            loss,
            optimizer,
            gradient_clip=adam_cfg["gradient_clip"],
            geometry=geometry,
        )
        adam_start = time.time()
        for local_step in range(1, adam_cfg["steps"] + 1):
            params, optimizer_state, total, components = train_step(
                params,
                optimizer_state,
                xx0,
                xx1,
                residual_equal_spin_sz,
                parity_xx0,
                parity_delta,
                parity_equal_spin_sz,
                outer_xx0,
                outer_xx1,
                outer_equal_spin_sz,
                source_terms[0],
                source_terms[1],
            )
            step = adam_start_step + local_step
            if local_step == 1 or local_step % adam_cfg["log_every"] == 0:
                values = _component_floats(components)
                print(
                    f"Adam {step}: total={float(total):.6e}, "
                    f"J_R_H={values['residual_loss']:.6e}, "
                    f"theta={values['theta_inner_parity_loss']:.6e}, "
                    f"Robin={values['outer_robin_boundary_loss']:.6e}",
                    flush=True,
                )
            if (
                adam_cfg["checkpoint_every"] > 0
                and local_step % adam_cfg["checkpoint_every"] == 0
                and local_step < adam_cfg["steps"]
            ):
                _save(
                    output / f"checkpoint_adam_{step:08d}.npz",
                    params,
                    cfg,
                    stage="adam",
                    step=step,
                    parent_hash=parent_hash,
                    total=total,
                    components=components,
                )
        print(f"Adam wall time: {time.time() - adam_start:.3f}s", flush=True)
        adam_final_step = adam_start_step + adam_cfg["steps"]
    else:
        adam_final_step = adam_start_step

    ssb_cfg = cfg["training"]["ssbroyden"]
    ssb_start_step = 0
    if resume_metadata is not None and resume_metadata["stage"] == "ssbroyden":
        ssb_start_step = int(resume_metadata["step"])
    ssb_final_step = ssb_start_step
    if ssb_cfg["steps"] > 0:
        if adam_cfg["steps"] > 0:
            _save(
                output / "checkpoint_adam_final.npz",
                params,
                cfg,
                stage="adam",
                step=adam_final_step,
                parent_hash=parent_hash,
                total=total,
                components=components,
            )
        from .ssbroyden_trainer import run_ssbroyden_stage

        def progress(iteration, current_loss, current_components):
            step = ssb_start_step + iteration
            print(
                f"SSBroyden {step}: total={current_loss:.6e}, "
                f"J_R_H={current_components['residual_loss']:.6e}, "
                f"theta={current_components['theta_inner_parity_loss']:.6e}, "
                f"Robin={current_components['outer_robin_boundary_loss']:.6e}",
                flush=True,
            )

        def save_ssb(iteration, best_params, _best_loss, _history):
            step = ssb_start_step + iteration
            checkpoint_total, checkpoint_components = loss.compute_loss(
                best_params, xx0, xx1, residual_equal_spin_sz, **loss_arguments
            )
            _save(
                output / f"checkpoint_ssbroyden_{step:08d}.npz",
                best_params,
                cfg,
                stage="ssbroyden",
                step=step,
                parent_hash=parent_hash,
                total=checkpoint_total,
                components=checkpoint_components,
            )

        (
            params,
            _history,
            total,
            component_values,
            elapsed,
            iterations,
            _best_loss,
            result,
        ) = run_ssbroyden_stage(
            params,
            xx0,
            xx1,
            residual_equal_spin_sz,
            parity_xx0,
            parity_delta,
            parity_equal_spin_sz,
            outer_xx0,
            outer_xx1,
            outer_equal_spin_sz,
            max_steps=ssb_cfg["steps"],
            progress_callback=progress,
            checkpoint_callback=save_ssb,
            checkpoint_step=ssb_cfg["checkpoint_every"],
            source_terms=source_terms,
            geometry=geometry,
        )
        components = {
            name: jnp.asarray(value) for name, value in component_values.items()
        }
        ssb_final_step = ssb_start_step + iterations
        print(
            f"SSBroyden finished after {iterations} steps in {elapsed:.3f}s: {result}",
            flush=True,
        )

    if ssb_cfg["steps"] > 0:
        final_stage, final_step = "ssbroyden", ssb_final_step
    elif adam_cfg["steps"] > 0:
        final_stage, final_step = "adam", adam_final_step
    else:
        final_stage = "resume" if resume_metadata is not None else "initialized"
        final_step = int(resume_metadata["step"]) if resume_metadata else 0
    final_path = output / "checkpoint_final.npz"
    _save(
        final_path,
        params,
        cfg,
        stage=final_stage,
        step=final_step,
        parent_hash=parent_hash,
        total=total,
        components=components,
    )
    print(f"Final loss: {float(total):.6e}", flush=True)
    return final_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--Nxx0", type=int)
    parser.add_argument("--Nxx1", type=int)
    parser.add_argument(
        "--equal-spin-sz-points",
        type=int,
        help="Number of cell centers for the raw equal Bowen-York S_z input.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--ssbroyden-steps", type=int)
    return parser.parse_args(argv)


def configured_run(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path | None]:
    cfg = config.load_config(args.config)
    overrides = {
        "Nxx0": args.Nxx0,
        "Nxx1": args.Nxx1,
        "equal_spin_sz_points": args.equal_spin_sz_points,
    }
    for name, value in overrides.items():
        if value is not None:
            cfg["collocation"][name] = value
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.adam_steps is not None:
        cfg["training"]["adam"]["steps"] = args.adam_steps
    if args.ssbroyden_steps is not None:
        cfg["training"]["ssbroyden"]["steps"] = args.ssbroyden_steps
    return config.validate_config(cfg), args.output_dir, args.resume_checkpoint


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg, output_dir, resume = configured_run(args)
    run_training(cfg, output_dir=output_dir, resume_checkpoint=resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
