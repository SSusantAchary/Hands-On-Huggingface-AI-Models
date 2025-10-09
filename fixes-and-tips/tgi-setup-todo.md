# TGI setup pending
**Symptom:** `docker: command not found` or `connection refused` when attempting to reach Text Generation Inference.
**Root cause:** TGI requires a running Docker container with GPU access; the local environment has not been provisioned.
**Fix:** Follow Hugging Face TGI docs to launch `ghcr.io/huggingface/text-generation-inference` with the desired model and expose the HTTP port.
**Verify:** `curl localhost:8080/health` returns `ok` and the notebook client receives responses.
**Scope:** CUDA GPUs, Docker, transformers>=4.40
**Related:** notebooks/serving/tgi-vs-pipeline-latency_microbenchmark.ipynb
