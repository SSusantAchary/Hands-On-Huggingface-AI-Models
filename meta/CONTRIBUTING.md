# Contributing

Thanks for helping modernise this repository. Follow the steps below to keep everything reproducible.

## Prerequisites
- Python ≥3.10 and `pip`.
- Optional: Apple Silicon (Metal) or CUDA GPU for accelerator paths.
- Install dependencies with `pip install -r notebooks/requirements-minimal.txt` and add optional packages as needed.

## Workflow
1. Fork and clone the repository.
2. Create a feature branch (`git checkout -b feature/my-update`).
3. Run or extend the relevant notebook, ensuring CPU execution works end-to-end.
4. Update measurements via the provided helper (`notebooks/_templates/measure.py`) and commit the modified `benchmarks/matrix.csv`.
5. Add or update fix entries if you solved a user-facing issue (see `/fixes-and-tips/` micro-template).
6. Run the CI checks locally if possible:
   - Markdown lint (markdownlint-cli or cspell)
   - Link check (lychee)
   - Notebook smoke test (`python -m nbclient run` or via `make smoke`)
7. Open a pull request using `.github/PULL_REQUEST_TEMPLATE.md`.

## Notebook Standards
See `meta/QUALITY.md` for the full definition of done. Highlights:
- Include licenses for models/datasets.
- Provide TL;DR, run profiles, Colab badge, and toggles at the top.
- Capture RAM/VRAM + throughput measurements after warmups.
- Summarise results and reproducibility details at the bottom.
- Reference relevant fixes from the Gotchas section.

## Fixes & Tips
- Create a new markdown file under `/fixes-and-tips/` using the provided micro-template.
- Keep the description short and link back to notebooks or PRs.
- Update the table in `docs/fixes-and-tips.md`.

## Benchmarks
- Use the helper to append rows to `benchmarks/matrix.csv`.
- Do not hand-edit existing measurements—re-run the notebook instead.
- Update highlight charts in `assets/benchmarks/` if the new data changes the story.

## Community Guidelines
- Be respectful and help reviewers by keeping PRs focused.
- Label issues/PRs with `notebook`, `fix`, `benchmark`, `apple-silicon`, `gpu`, `good-first-issue` where relevant.
