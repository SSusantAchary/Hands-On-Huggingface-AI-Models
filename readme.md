<div align="center">

<p><strong>Developer-first, CPU-friendly Hugging Face notebooks with transparent measurements.</strong></p>

<p>
  <a href="/docs/gallery.md" style="padding:12px 20px;margin:4px;display:inline-block;border-radius:8px;background:#1f6feb;color:#fff;text-decoration:none;font-weight:600;">Notebook Gallery</a>
  <a href="/docs/fixes-and-tips.md" style="padding:12px 20px;margin:4px;display:inline-block;border-radius:8px;background:#0d1117;color:#fff;text-decoration:none;font-weight:600;">Fixes &amp; Tips</a>
  <a href="/docs/benchmarks.md" style="padding:12px 20px;margin:4px;display:inline-block;border-radius:8px;background:#6f42c1;color:#fff;text-decoration:none;font-weight:600;">Benchmark Highlights</a>
</p>

[![GitHub Repo stars](https://img.shields.io/github/stars/SSusantAchary/Hands-On-Huggingface-AI-Models?style=social)](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Built with 🤗 Transformers](https://img.shields.io/badge/Built%20with-%F0%9F%A4%97%20Transformers-ff4a9c.svg)](https://huggingface.co/transformers)

</div>

---

**This week** (see `meta/CHANGELOG.md` for details):
- Gallery kickoff with CPU-first sentiment, ViT Imagenette, and Whisper notebooks.
- Fix of the Week: Metal backend fallback detects missing MPS and guides to CPU.
- Mini-benchmark of the Week: CLIP batch-size sweep (tokens/s placeholder until run).

---

### Run Anywhere Pledge
- CPU is the default path; every notebook runs end-to-end without a GPU.
- Optional toggles enable Apple Silicon (Metal) or CUDA acceleration when present.
- Measurements list RAM/VRAM footprints, first-token latency, and throughput per run.

### What's Inside
- Curated notebook gallery across NLP, vision, audio, multimodal, and serving tasks.
- Single-source benchmark matrix (`benchmarks/matrix.csv`) + lightweight charts.
- Fixes & Tips index (symptom → fix → verify → scope) for common Hugging Face hurdles.
- MkDocs-powered docs site for quick navigation and sharing.

### Quick Start
```bash
git clone https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models.git
cd Hands-On-Huggingface-AI-Models
python -m venv .venv && source .venv/bin/activate
pip install -r notebooks/requirements-minimal.txt
```
Open any notebook from `/notebooks` in Jupyter, VS Code, or Colab (badges inside each file).

### Who It's For
- Practitioners who need reproducible, CPU-safe Hugging Face experiments.
- Teams validating Metal or CUDA paths without breaking CPU workflows.
- Contributors adding benchmarks, fixes, or doc polish with low friction.

**Prereqs:** Python ≥3.10, git, and `pip`. GPU/Metal optional.

### Fix of the Week
- Metal backend fallback detection with CPU guidance – documented in `/fixes-and-tips/metal-backend-fallback.md`.

### Mini-benchmark of the Week
- CLIP retrieval batch-size sweep scaffolding – see `/docs/benchmarks.md` and run the notebook to populate metrics.

### Cite & License
```
@misc{hands-on-hf,
  author    = {S. Susant Achary},
  title     = {Hands-On Hugging Face AI Models},
  year      = {2025},
  howpublished = {\url{https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models}}
}
```
Licensed under the [MIT License](LICENSE).

---

Questions or ideas? Open an issue with labels `notebook`, `fix`, `benchmark`, or `apple-silicon` to help us triage fast.
