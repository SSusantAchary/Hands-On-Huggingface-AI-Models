# Notebook Definition of Done

Use this checklist before opening a pull request.

- [ ] CPU-only run passes end-to-end without manual patching.
- [ ] RAM + (if applicable) VRAM peaks recorded in the measurement cell.
- [ ] Throughput + latency values appended to `benchmarks/matrix.csv`.
- [ ] Notebook top cell includes TL;DR, run profiles, licenses, Colab badge.
- [ ] Gotchas section populated with link into `/fixes-and-tips/`.
- [ ] Results summary & reproducibility footer (seed, versions, commit hash) filled in.
- [ ] Optional accelerators (Metal/CUDA) guarded by capability checks.
- [ ] Outputs saved under `/outputs` or in-memory only; no large artefacts committed.
