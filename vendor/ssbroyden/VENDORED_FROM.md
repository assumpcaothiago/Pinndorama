# SSBroyden and Optimistix provenance

## Upstream sources

- SSBroyden repository:
  `https://github.com/IvanBioli/ssbroyden_optimistix`
- SSBroyden commit: `4c87785c68f0fec6b09000f474daef76fb181eea`
- Retrieval date: 2026-07-10
- Optimistix fork repository: `https://github.com/IvanBioli/optimistix`
- Optimistix submodule commit:
  `8cd4931713658f8dfe4423ead6f11b348b675540`
- Optimistix version declared by that snapshot: `0.1.0`

The active vendor tree contains ordinary files rather than gitlinks. It is
therefore sufficient to run the optimizer if either upstream repository
becomes unavailable. Upstream examples, figures, repository tooling, and
unrelated tests are intentionally excluded from the publication runtime.

## Pinndorama adaptations

- Upstream's `ssbrodyen_family.py` path is normalized to
  `ssbroyden_family.py`; its contents are unchanged.
- Pinndorama commit `00105800b084006216532cbcea36c7c996fe5f6d`
  modified `optimistix_wrapper.py` to reuse the accepted state's stored loss
  for callbacks and final reporting instead of reevaluating the complete PINN
  loss.
- `optimistix/__init__.py` reports the pinned constant version `0.1.0`
  instead of querying separately installed distribution metadata. This is a
  packaging-only change and does not alter optimization mathematics.
- Pinndorama commit `d59ebb5` applied Black 26.5.1 formatting to 15 nested
  Optimistix Python files. This formatting-only change does not alter the
  optimizer algorithm; `SHA256SUMS` records the formatted source.
- Three historically separate, numerically identical runtime copies were
  consolidated into this single shared snapshot for all publication solvers.

`SHA256SUMS` records the resulting active runtime. Any source change requires
updating both this provenance document and the checksum manifest, followed by
all three float64 SSBroyden smoke tests.

## License and citation

The nested `optimistix/LICENSE` is the Apache License 2.0 and applies to the
nested Optimistix snapshot. No separate license was present for the SSBroyden
add-on files at the pinned upstream commit. Written redistribution permission
is pending; this document does not infer or assign a license to those files.

Please cite the upstream technical note:

```bibtex
@misc{bioli2026selfscaledbroydenfamilyquasinewton,
  title={Self-Scaled Broyden Family of Quasi-Newton Methods in JAX},
  author={Ivan Bioli and Mikel Mendibe Abarrategi},
  year={2026},
  eprint={2603.10599},
  archivePrefix={arXiv},
  primaryClass={cs.MS},
  url={https://arxiv.org/abs/2603.10599}
}
```
