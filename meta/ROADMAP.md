# 4-Week Roadmap

## Week 1
- Notebook: Finalise IMDB sentiment fine-tune path (LoRA stub → runnable demo).
- Fix: Document `bitsandbytes` CPU fallback workflow.
- Mini-benchmark: Measure CLIP retrieval batch size 1–16 on CPU + Metal.

## Week 2
- Notebook: Ship Imagenette ViT fine-tune with accuracy + confusion matrix outputs.
- Fix: Add troubleshooting guide for Whisper audio dependencies on macOS.
- Mini-benchmark: Capture Whisper Tiny CPU vs Metal throughput on 15s clips.

## Week 3
- Notebook: Expand BLIP captioning with Flickr8k evaluation cache.
- Fix: Torch compile + MPS guardrails for vision models.
- Mini-benchmark: TGI vs pipeline latency baseline (CPU stub + TODO for GPU).

## Week 4
- Notebook: FastAPI serving notebook → deployable script + Docker hints.
- Fix: Colab CUDA runtime mismatch auto-detect.
- Mini-benchmark: Summarisation (T5-small) throughput for context lengths 256/512.
