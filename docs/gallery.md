# Notebook Gallery

Every notebook is CPU-safe by default, ships with Metal/CUDA toggles, and appends measurements to `benchmarks/matrix.csv` once you run the auto-measure cell.

## NLP

### Sentiment Analysis — DistilBERT on IMDB
- **Task:** Sentiment classification pipeline with optional LoRA fine-tune stub.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/nlp/sentiment-distilbert-imdb_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/input-tokenizer-oom.md) · [Model Card](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/nlp/sentiment-distilbert-imdb_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

### Summarization — T5-small on CNN/DM
- **Task:** Abstractive summarisation with evaluation via ROUGE.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/nlp/summarization-t5-small-cnn_dm_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/tokenizer-context-oom.md) · [Model Card](https://huggingface.co/t5-small) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/nlp/summarization-t5-small-cnn_dm_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

### Instruction Generation — Llama 3 Instruct 8B Lite
- **Task:** Toy instruction following with CPU-first setup and TODO model gate.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/nlp/instruct-generation-llama-3-instruct-8b_lite_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/bitsandbytes-wheel-mismatch.md) · [Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/nlp/instruct-generation-llama-3-instruct-8b_lite_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

## Vision

### Classification — ViT Base on Imagenette
- **Task:** Zero-shot vs. fine-tuned comparison on Imagenette.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/vision/classification-vit-base-224-imagenette_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/torch-compile-mps-quirks.md) · [Model Card](https://huggingface.co/google/vit-base-patch16-224) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/vision/classification-vit-base-224-imagenette_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

### Zero-shot Retrieval — CLIP
- **Task:** Text-image retrieval with cosine similarity table.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/vision/zero-shot-clip-retrieval_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/metal-backend-fallback.md) · [Model Card](https://huggingface.co/openai/clip-vit-base-patch32) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/vision/zero-shot-clip-retrieval_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

### Detection — DETR ResNet-50
- **Task:** Object detection on sample images with overlay visualisations.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/vision/detection-detr-resnet50_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/torch-compile-mps-quirks.md) · [Model Card](https://huggingface.co/facebook/detr-resnet-50) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/vision/detection-detr-resnet50_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

## Audio

### Speech-to-Text — Whisper Tiny
- **Task:** Transcribe short clips and report WER (toy).
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/audio/asr-whisper-tiny-base_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/colab-cuda-mismatch.md) · [Model Card](https://huggingface.co/openai/whisper-tiny) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/audio/asr-whisper-tiny-base_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

### Audio Classification — HuBERT SUPERB
- **Task:** Audio classification with accuracy summary.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/audio/audio-classification-hubert-superb_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/input-tokenizer-oom.md) · [Model Card](https://huggingface.co/superb/hubert-base-superb-ks) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/audio/audio-classification-hubert-superb_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

## Multimodal

### Captioning — BLIP Base on Flickr8k
- **Task:** Generate captions with BLEU (toy) evaluation.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/multimodal/captioning-blip-base_flickr8k_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/tokenizer-context-oom.md) · [Model Card](https://huggingface.co/Salesforce/blip-image-captioning-base) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/multimodal/captioning-blip-base_flickr8k_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

### Retrieval Study — CLIP Mini Batch Effects
- **Task:** Measure throughput impact of batch size choices.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/multimodal/retrieval-clip-mini-batch_effects_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/metal-backend-fallback.md) · [Model Card](https://huggingface.co/openai/clip-vit-base-patch32) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/multimodal/retrieval-clip-mini-batch_effects_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

## Serving

### FastAPI Pipeline Demo
- **Task:** Minimal FastAPI wrapper for text generation.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/serving/serving-fastapi-pipeline_demo_cpu-first.ipynb) · [Fix entry](../fixes-and-tips/fastapi-local-run.md) · [Model Card](https://huggingface.co/distilbert-base-uncased) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/serving/serving-fastapi-pipeline_demo_cpu-first.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO

### TGI vs Pipeline Latency Microbenchmark
- **Task:** Conceptual comparison with TODO placeholders until TGI is available.
- **Run Profiles:** 🖥️ CPU | 🍎 Metal | 🧪 Colab/T4 | ⚡ CUDA
- **Open:** [Notebook](https://github.com/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/serving/tgi-vs-pipeline-latency_microbenchmark.ipynb) · [Fix entry](../fixes-and-tips/tgi-setup-todo.md) · [Model Card](https://huggingface.co/docs/text-generation-inference/index) · [Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/notebooks/serving/tgi-vs-pipeline-latency_microbenchmark.ipynb)
- **Mini-metrics:** RAM — TODO | Throughput — TODO | Quality — TODO
