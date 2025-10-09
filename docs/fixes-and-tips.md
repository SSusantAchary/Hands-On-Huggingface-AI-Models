# Fixes & Tips Index

Use the table below to jump straight to a symptom and apply the ready-to-run fix.

| Symptom | Cause | Fix link | Affects |
|---|---|---|---|
| Metal backend fallback | MPS device missing or not initialized | ./metal-backend-fallback.md | Apple Silicon |
| bitsandbytes wheel mismatch | Platform-specific wheel unavailable | ./bitsandbytes-wheel-mismatch.md | CUDA GPU |
| Tokenizer OOM for long context | Sequence length exceeds memory | ./tokenizer-context-oom.md | CPU / GPU |
| Torch compile quirks on MPS | PyTorch compile path unsupported on MPS | ./torch-compile-mps-quirks.md | Apple Silicon |
| Colab CUDA mismatch | CUDA driver/runtime conflict in Colab | ./colab-cuda-mismatch.md | Colab / CUDA |
| Input tokenizer OOM on audio | Audio-to-text token expansion spikes RAM | ./input-tokenizer-oom.md | CPU / GPU |
| FastAPI local run errors | Uvicorn reload clashing with notebook event loop | ./fastapi-local-run.md | Serving |
| TGI setup TODO | Need infra before TGI benchmarks run | ./tgi-setup-todo.md | Serving / Benchmarks |
