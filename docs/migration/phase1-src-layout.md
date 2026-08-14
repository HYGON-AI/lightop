# Phase 1 Python Package Skeleton

## Layout

- Installable Python sources live under `python/lightop/`.
- Native C++/HIP sources live under the repository-level `csrc/`.
- Runtime resources remain in repository-level `configs/`, `hsa/`, and
  `lib/` directories.
- Legacy Python modules remain in `python/lightop/` while the new functional
  domain packages are introduced as empty API skeletons.
- `sampling/` and `utils/` are placeholders without `__init__.py` in this
  phase because package directories would shadow the legacy `sampling.py` and
  `utils.py` modules.

## Build behavior

- `setup.py` discovers packages with `find_packages(where="python")` and
  `package_dir={"": "python"}`.
- The compiled extension remains named `lightop.op`.
- A custom `build_py` command copies repository-level runtime resources into
  the installed `lightop` package.
- Source-checkout resource lookup falls back from `python/lightop/` to the
  repository root.
- `csrc/allreduce_fuse/` remains outside the extension source list and wheel.
