# Pinned shared SSBroyden runtime

This directory is Pinndorama's self-contained runtime snapshot of the
Self-Scaled Broyden add-on and its required Optimistix fork. All publication
solvers load these same files after their Adam stage; no clone, submodule
checkout, or runtime download is required.

- `ssbroyden_family.py` implements the self-scaled Broyden family.
- `optimistix_wrapper.py` provides the accepted-step optimization loop used by
  Pinndorama.
- `optimistix/` contains the complete required Optimistix Python runtime,
  packaging metadata, README, `py.typed`, and Apache-2.0 license.

Pinndorama prepends this directory to the import path and rejects Optimistix,
wrapper, or family modules loaded from anywhere else. See `VENDORED_FROM.md`
for upstream commits, local patches, citation, and license scope. `SHA256SUMS`
records every active runtime and dependency file.
