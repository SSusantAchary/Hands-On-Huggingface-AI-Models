# Tokenizer OOM for long context
**Symptom:** `RuntimeError: CUDA out of memory` or `Killed: 9` while tokenising long documents.
**Root cause:** Inputs exceed the model's maximum context length, causing tokenisers to allocate large temporary buffers.
**Fix:** Truncate or chunk inputs with `max_length`/`truncation=True`, optionally summarising in rounds.
**Verify:** Re-run the summarisation notebook; tokenisation should complete with memory usage under control.
**Scope:** transformers>=4.40, datasets>=2.18, CPU/CUDA/Metal
**Related:** notebooks/nlp/summarization-t5-small-cnn_dm_cpu-first.ipynb, notebooks/multimodal/captioning-blip-base_flickr8k_cpu-first.ipynb
