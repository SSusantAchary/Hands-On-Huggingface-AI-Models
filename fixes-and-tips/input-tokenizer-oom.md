# Audio tokenizer RAM spike
**Symptom:** `Killed: 9` or `MemoryError` while converting audio waveforms to tokens.
**Root cause:** Large raw audio arrays (sample_rate * duration) exhaust system RAM during feature extraction.
**Fix:** Downsample to 16 kHz, limit clip duration, and process batches incrementally instead of loading all clips.
**Verify:** Audio classification notebook runs through without being terminated by the OS.
**Scope:** transformers>=4.40, datasets>=2.18, CPU/GPU
**Related:** notebooks/audio/audio-classification-hubert-superb_cpu-first.ipynb
