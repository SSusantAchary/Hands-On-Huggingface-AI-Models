# Colab CUDA mismatch
**Symptom:** `CUDA driver version is insufficient for CUDA runtime version` in Google Colab.
**Root cause:** Colab occasionally pins NVIDIA drivers behind the CUDA runtime expected by recent PyTorch wheels.
**Fix:** Select the matching Colab runtime (e.g. "RunTime → Change runtime type → GPU (T4)") and install `torch==2.3.1` plus matching `nvidia-cuda-runtime-cu118` wheels.
**Verify:** `torch.cuda.is_available()` returns `True` and notebooks import without warnings.
**Scope:** Google Colab, CUDA GPUs, torch>=2.1
**Related:** notebooks/audio/asr-whisper-tiny-base_cpu-first.ipynb
