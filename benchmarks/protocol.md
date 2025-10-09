# Benchmark Protocol

1. **Warmups:** Run each workload twice before recording metrics to warm caches and kernels.
2. **Seed:** Set `torch.manual_seed(42)` (and equivalent for numpy/random) before inference or training.
3. **Environment:** Capture CPU/GPU/Metal model, OS, and library versions via `detect_env()` (already invoked inside `append_benchmark_row`).
4. **Workload:** Single prompt / single sample unless the notebook explicitly studies batching. Note the sequence length or image resolution used.
5. **Precision:** Record precision/quantisation mode (`fp32`, `fp16`, `bf16`, `int8`, `4bit`, etc.).
6. **Metrics:** Record load time, peak RAM, peak VRAM (if applicable), first-token latency (TTFB), and throughput (tokens/s or images/s).
7. **Reporting:** Append the row using `append_benchmark_row(...)` and commit the updated CSV. Do not edit prior rows manually.
8. **Artefacts:** Keep generated artefacts lightweight (tables/plots under `assets/benchmarks/`).
