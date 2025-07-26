# ✨ Small Language Model Repository

Welcome to the **Small Language Model Repository**! This project focuses on building lightweight, domain-specific language models for diverse applications, including multilingual text, Video, audio-based processing and more.

---

### Overview of Popular Open Models and Formats

This table summarizes leading open-source models with less than 10 billion parameters, focusing on their availability across different optimized formats.

*   **MLX**: For efficient inference on Apple Silicon (M-series chips).
*   **Unsloth FT (4-bit)**: Optimized for significantly faster fine-tuning with less memory, using 4-bit quantization.
*   **Unsloth GGUF**: Quantized for fast CPU-based inference and compatibility with tools like Ollama and Llama.cpp.

| Model | Parameter Size | Tasks | Base Version | Fine-tuned / Instruct | MLX (Apple Silicon) | Unsloth FT (4-bit) | Unsloth GGUF | License Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3** | 8B | Text Gen, Chat, Code | [`meta-llama/Llama-3-8B`](https://huggingface.co/meta-llama/Llama-3-8B) | [`meta-llama/Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | [`mlx-community/Meta-Llama-3-8B-Instruct-mlx`](https://huggingface.co/mlx-community/Meta-Llama-3-8B-Instruct-mlx) | [`unsloth/llama-3-8b-Instruct-bnb-4bit`](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit) | [`unsloth/llama-3-8b-Instruct-GGUF`](https://huggingface.co/unsloth/llama-3-8b-Instruct-GGUF) | Llama 3 Community |
| **Gemma** | 7B | Text Gen, Chat, Reasoning | [`google/gemma-7b`](https://huggingface.co/google/gemma-7b) | [`google/gemma-7b-it`](https://huggingface.co/google/gemma-7b-it) | [`mlx-community/gemma-7b-it-mlx`](https://huggingface.co/mlx-community/gemma-7b-it-mlx) | [`unsloth/gemma-7b-it-bnb-4bit`](https://huggingface.co/unsloth/gemma-7b-it-bnb-4bit) | [`unsloth/gemma-7b-it-GGUF`](https://huggingface.co/unsloth/gemma-7b-it-GGUF) | Gemma Terms of Use |
| **Mistral** | 7B | Text Gen, Function Calling | [`mistralai/Mistral-7B-v0.3`](https://huggingface.co/mistralai/Mistral-7B-v0.3) | [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | [`mlx-community/Mistral-7B-Instruct-v0.3-mlx`](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-mlx) | [`unsloth/mistral-7b-instruct-v0.3-bnb-4bit`](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit) | [`unsloth/mistral-7b-instruct-v0.3-GGUF`](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-GGUF) | Apache 2.0 |
| **Qwen2** | 7B | Chat, Multilingual, Code | [`Qwen/Qwen2-7B`](https://huggingface.co/Qwen/Qwen2-7B) | [`Qwen/Qwen2-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | [`mlx-community/Qwen2-7B-Instruct-mlx`](https://huggingface.co/mlx-community/Qwen2-7B-Instruct-mlx) | [`unsloth/Qwen2-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2-7B-Instruct-bnb-4bit) | [`unsloth/Qwen2-7B-Instruct-GGUF`](https://huggingface.co/unsloth/Qwen2-7B-Instruct-GGUF) | Apache 2.0 |
| **Phi-3** | 3.8B | Text Gen, Chat, Reasoning | N/A | [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | [`mlx-community/Phi-3-mini-4k-instruct-mlx`](https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-mlx) | [`unsloth/Phi-3-mini-4k-instruct-bnb-4bit`](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit) | [`unsloth/Phi-3-mini-4k-instruct-GGUF`](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-GGUF) | MIT |
| **Llama 2** | 7B | Text Gen, Chat | [`meta-llama/Llama-2-7b`](https://huggingface.co/meta-llama/Llama-2-7b) | [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [`mlx-community/Llama-2-7b-chat-mlx`](https://huggingface.co/mlx-community/Llama-2-7b-chat-mlx) | [`unsloth/llama-2-7b-chat-bnb-4bit`](https://huggingface.co/unsloth/llama-2-7b-chat-bnb-4bit) | [`unsloth/llama-2-7b-chat-GGUF`](https://huggingface.co/unsloth/llama-2-7b-chat-GGUF) | Llama 2 Community |
| **TinyLlama**| 1.1B | Chat, Text Gen | [`TinyLlama/...-step-1431k-3T`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) | [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | [`mlx-community/TinyLlama-1.1B-Chat-v1.0-mlx`](https://huggingface.co/mlx-community/TinyLlama-1.1B-Chat-v1.0-mlx) | [`unsloth/tinyllama-bnb-4bit`](https://huggingface.co/unsloth/tinyllama-bnb-4bit) | [`unsloth/tinyllama-GGUF`](https://huggingface.co/unsloth/tinyllama-GGUF) | Apache 2.0 |


## 🎧 List of Audio Language Models

Below are 10 popular audio language models with their licensing details:

| 🎙️ **Model**            | 🌐 **Description**                                                                                      | 📜 **License**      | 🔗 **More Info**                             |
|--------------------------|------------------------------------------------------------------------------------------------------|---------------------|---------------------------------------------|
| **1. Whisper by OpenAI** 🧠 | Robust speech recognition model supporting multilingual transcription and translation.                  | MIT License         | [GitHub](https://github.com/openai/whisper) |
| **2. Wav2Vec 2.0 by Facebook AI** 🌊 | Self-supervised learning for speech recognition with outstanding performance on ASR tasks.          | Apache 2.0          | [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) |
| **3. DeepSpeech by Mozilla** 🔊 | Open-source speech-to-text model inspired by Baidu's DeepSpeech research.                          | MPL 2.0             | [GitHub](https://github.com/mozilla/DeepSpeech) |
| **4. Kaldi** 🎤            | Toolkit for speech recognition with support for advanced algorithms and flexible architectures.      | Apache 2.0          | [Website](https://kaldi-asr.org/)           |
| **5. SpeechBrain** 🧠     | All-in-one toolkit for speech technologies, including ASR, speaker recognition, and enhancement.     | Apache 2.0          | [GitHub](https://github.com/speechbrain/speechbrain) |
| **6. Coqui STT** 🐸        | TTS and ASR model forked from Mozilla DeepSpeech, focusing on speed and ease of use.                 | MPL 2.0             | [GitHub](https://github.com/coqui-ai/STT)   |
| **7. OpenSLR** 🗃️         | Repository of open speech and language resources for building speech models.                        | Various (depends on resource) | [Website](http://www.openslr.org/)          |
| **8. PaddleSpeech** 🐼     | End-to-end speech toolkit by Baidu focusing on ASR, TTS, and other speech-related tasks.            | Apache 2.0          | [GitHub](https://github.com/PaddlePaddle/PaddleSpeech) |
| **9. ESPnet** 🚀          | End-to-end speech processing toolkit for ASR, TTS, and STT with modern neural architectures.         | Apache 2.0          | [GitHub](https://github.com/espnet/espnet)  |
| **10. Jasper by NVIDIA** ⚡ | End-to-end ASR model optimized for GPUs, focusing on accuracy and training speed.                   | Apache 2.0          | [GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper) |

---
## 🎯 Key Tasks Supported by Audio Language Models

Audio Language Models play a vital role in processing and understanding human speech. Below are the key tasks these models support:

### 1. **📝 Automatic Speech Recognition (ASR)**
   - Transcribes spoken words into written text.
   - Applications: Voice assistants, transcription services, and real-time captioning.

### 2. **🌍 Multilingual Speech Recognition**
   - Recognizes and transcribes speech in multiple languages.
   - Applications: Global communication, language learning tools, and multilingual customer support.

### 3. **🔄 Speech Translation**
   - Translates spoken language into another language in real-time or offline.
   - Applications: Cross-language communication, travel assistants, and international conferences.

### 4. **🗣️ Text-to-Speech (TTS) Conversion**
   - Converts written text into human-like speech.
   - Applications: Audiobooks, accessibility tools for visually impaired users, and voiceovers.

### 5. **👤 Speaker Identification**
   - Identifies individual speakers from their voice.
   - Applications: Voice biometrics, security systems, and personalized voice assistants.

### 6. **👥 Speaker Diarization**
   - Distinguishes and labels different speakers in an audio recording.
   - Applications: Meeting transcription, multi-speaker podcasts, and conference recordings.

### 7. **🔊 Emotion Recognition**
   - Detects emotions from speech signals.
   - Applications: Call center analysis, mental health monitoring, and gaming interactions.

### 8. **🎧 Audio Enhancement**
   - Improves the quality of audio signals by removing noise or improving clarity.
   - Applications: Teleconferencing, podcast production, and hearing aids.

### 9. **🔍 Keyword Spotting**
   - Detects specific keywords or phrases in audio streams.
   - Applications: Wake-word detection (e.g., "Hey Siri"), surveillance, and targeted audio analysis.

### 10. **🧠 Sentiment Analysis from Speech**
   - Analyzes the sentiment (positive, neutral, or negative) conveyed through speech.
   - Applications: Customer feedback analysis, social interactions, and content moderation.

### 11. **📊 Speech Analytics**
   - Extracts insights such as word usage, speech pace, and patterns from audio data.
   - Applications: Business analytics, interview analysis, and training evaluations.

### 12. **🎮 Voice Control for Devices**
   - Enables interaction with devices or systems using voice commands.
   - Applications: Smart home devices, gaming controls, and industrial automation.

---

These tasks demonstrate the versatility and importance of audio language models across industries. Whether for accessibility, communication, or business applications, these models transform how we interact with technology and audio data.
