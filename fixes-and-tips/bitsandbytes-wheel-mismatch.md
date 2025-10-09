# bitsandbytes wheel mismatch
**Symptom:** `RuntimeError: No GPU detected. Please compile bitsandbytes with the right compute capability.`
**Root cause:** bitsandbytes only ships wheels for NVIDIA GPUs; installing it on CPU or Metal-only machines triggers the mismatch.
**Fix:** Skip bitsandbytes for CPU/Metal runs and rely on full-precision weights, or install the CPU-only fork (`pip install bitsandbytes-cpu`).
**Verify:** Re-run the notebook import cell; bitsandbytes should not be imported on CPU paths.
**Scope:** transformers>=4.40, accelerate>=0.29, bitsandbytes 0.43
**Related:** notebooks/nlp/instruct-generation-llama-3-instruct-8b_lite_cpu-first.ipynb
