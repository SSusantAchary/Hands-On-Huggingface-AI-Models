## 🧩 Hugging Face Ecosystem — Dev Quick Picks
<sub>Last reviewed: 2025-10-08</sub>
<br>
<sup>Opinionated, regularly used pieces that pair well with 🤗. Links go to official docs where possible.</sup>



This page is a curated list of developer-first libraries that integrate well with Hugging Face for training, optimization, serving, data, and UI. We keep it short and practical—if it’s not used in anger monthly, we leave it out.



| Category | Library | What it’s for | Where it fits |
|---|---|---|---|
| Core | [Transformers](https://huggingface.co/docs/transformers/index) | SOTA models across text/vision/audio/multimodal; training & inference | Model APIs, Trainer, generate() |
|  | [Datasets](https://huggingface.co/docs/datasets) | Stream/load/share datasets at scale | Data I/O, preprocessing, splits |
|  | [Tokenizers](https://huggingface.co/docs/tokenizers/index) | Fast Rust-backed tokenizers | Prod tokenization, custom vocab |
|  | [Evaluate](https://huggingface.co/docs/evaluate/index) | Metrics & eval pipelines | Reproducible metrics |
|  | [Diffusers](https://huggingface.co/docs/diffusers/index) | Diffusion models for images/video/audio | GenAI imaging/video |
| Training & Post-Training | [Accelerate](https://huggingface.co/docs/accelerate/index) | Device/distributed orchestration | Multi-GPU/TPU/MPS launch |
|  | [PEFT](https://huggingface.co/docs/peft/index) | LoRA/QLoRA & adapter training | Parameter-efficient finetuning |
|  | [TRL](https://huggingface.co/docs/trl/index) | SFT, DPO, PPO, RM for LLMs | Post-training & alignment |
|  | [DeepSpeed](https://www.deepspeed.ai/) | ZeRO + memory-efficient training | Large-model training via Trainer |
|  | [Optimum](https://huggingface.co/docs/optimum/index) | Hardware-specific speedups | ONNX, Intel, NVIDIA, AWS Neuron |
| Quant & Memory | [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 8-bit / 4-bit loading & QLoRA | Low-VRAM inference/finetune |
| Serving & Deployment | [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) | High-perf LLM serving | Prod text-gen endpoints |
|  | [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) | High-perf embedding serving | Retrieval/semantic search |
|  | [huggingface_hub](https://huggingface.co/docs/huggingface_hub) | Programmatic Hub client | Push/pull models, datasets, Spaces |
| Hub, DataFrames & ETL | [Hub integrations (Polars/Pandas/DuckDB/Dask/Spark/Daft)](https://huggingface.co/docs/hub/en/extensions) | Write/read to Hub repos from tables | Dataset pipelines & exports |
| Apps & Demos | [Spaces + Gradio](https://huggingface.co/docs/hub/en/spaces-sdks-gradio) | Share interactive demos on the Hub | UI for notebooks & models |



_Maintenance: Source of truth → `/meta/ecosystem.yml`._

