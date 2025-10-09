# Good First Issues Backlog

1. **Notebook metrics:** Run `notebooks/nlp/sentiment-distilbert-imdb_cpu-first.ipynb` on CPU and push filled RAM/throughput metrics to `benchmarks/matrix.csv`.
2. **Notebook metrics:** Execute `notebooks/vision/classification-vit-base-224-imagenette_cpu-first.ipynb` on Apple Silicon (MPS) and add measurements.
3. **Notebook metrics:** Benchmark `notebooks/audio/asr-whisper-tiny-base_cpu-first.ipynb` on CUDA (if available) and update the CSV.
4. **Fix write-up:** Document a troubleshooting guide for Whisper dependencies on macOS (brew `ffmpeg`, `soundfile`).
5. **Fix update:** Add a fix entry for Hugging Face Hub rate-limit handling with `HF_HUB_ENABLE_HF_TRANSFER=1`.
6. **Serving enhancement:** Extend `serving-fastapi-pipeline_demo_cpu-first.ipynb` with a Dockerfile snippet and smoke test in CI.
