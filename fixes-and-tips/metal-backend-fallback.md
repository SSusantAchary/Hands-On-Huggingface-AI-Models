# Metal backend fallback
**Symptom:** `RuntimeError: MPS backend is not available; falling back to CPU.`
**Root cause:** PyTorch detects Apple Silicon hardware but Metal drivers or the `PYTORCH_ENABLE_MPS_FALLBACK` flag are missing.
**Fix:** Guard device selection with `torch.backends.mps.is_available()` and default to CPU when false.
**Verify:** Run any notebook cell that prints `Using device=` — it should report `cpu` instead of erroring.
**Scope:** torch>=2.1, transformers>=4.40, Apple Silicon
**Related:** notebooks/vision/zero-shot-clip-retrieval_cpu-first.ipynb, notebooks/multimodal/retrieval-clip-mini-batch_effects_cpu-first.ipynb
