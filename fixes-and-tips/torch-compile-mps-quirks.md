# torch.compile with MPS quirks
**Symptom:** `NotImplementedError: The operator aten::scaled_dot_product_efficient_attention is not currently supported on MPS.`
**Root cause:** `torch.compile` still lacks coverage for several attention kernels on the Metal backend.
**Fix:** Skip `torch.compile` when `torch.backends.mps.is_available()`; run eager mode for Metal and re-enable on CUDA.
**Verify:** Execute the vision notebook setup cell; it should proceed without compile errors.
**Scope:** torch 2.1–2.4, Apple Silicon (MPS)
**Related:** notebooks/vision/classification-vit-base-224-imagenette_cpu-first.ipynb, notebooks/vision/detection-detr-resnet50_cpu-first.ipynb
