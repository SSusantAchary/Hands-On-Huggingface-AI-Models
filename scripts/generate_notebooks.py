from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]

BASE_ENV = "python>=3.10, transformers, datasets, evaluate, accelerate (optional: peft, bitsandbytes, timm, diffusers)"
RUN_PROFILES = "🖥️ CPU | 🍎 Metal (Apple Silicon) | 🧪 Colab/T4 | ⚡ CUDA GPU"
SWITCHES = "\n".join(
    [
        '- `device` = {"cpu","mps","cuda"}',
        '- `precision` = {"fp32","fp16","bf16","int8","4bit"}  (apply only if supported)',
        "- `context_len` / `image_res` / `batch_size`",
    ]
)


class NotebookSpec(dict):
    path: Path
    title: str
    task_subtitle: str
    tldr: str
    model_link: str
    model_name: str
    model_license: str
    dataset_link: str
    dataset_name: str
    dataset_license: str
    gotchas_link: str
    gotchas_label: str
    colab_path: str
    body_cells: List[dict]


NOTEBOOKS: List[NotebookSpec] = [
    NotebookSpec(
        path=Path("notebooks/nlp/sentiment-distilbert-imdb_cpu-first.ipynb"),
        title="Sentiment Analysis",
        task_subtitle="DistilBERT on IMDB",
        tldr="Classify IMDB movie reviews with a CPU-first DistilBERT pipeline and prep for LoRA fine-tuning.",
        model_link="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english",
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        model_license="Apache-2.0",
        dataset_link="https://huggingface.co/datasets/imdb",
        dataset_name="IMDB",
        dataset_license="CC BY-NC 4.0",
        gotchas_link="../fixes-and-tips/metal-backend-fallback.md",
        gotchas_label="Metal backend falls back to CPU if MPS unavailable",
        colab_path="notebooks/nlp/sentiment-distilbert-imdb_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Configure device toggles, load a small IMDB slice, and prepare utility helpers.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import pandas as pd
                    import torch
                    from datasets import load_dataset
                    from evaluate import load as load_metric
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")
                    BATCH_SIZE = int(os.environ.get("HF_BATCH", "4"))

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE} (precision={PRECISION})")

                    DATASET_ID = "imdb"
                    MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
                    OUTPUT_DIR = Path("outputs") / "sentiment-distilbert-imdb"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    dataset = load_dataset(DATASET_ID, split="test[:16]")
                    sample = dataset.shuffle(seed=42).select(range(BATCH_SIZE))
                    texts = sample["text"]
                    labels = sample["label"]

                    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
                    print(f"Loaded {len(texts)} samples for smoke run.")
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Inference & Evaluation",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    classifier = pipeline(
                        "text-classification",
                        model=MODEL_ID,
                        device=DEVICE,
                        top_k=None,
                        padding=True,
                        truncation=True,
                        batch_size=BATCH_SIZE,
                        return_all_scores=True,
                    )
                    load_time = time.perf_counter() - load_start

                    all_scores = classifier(texts)
                    predictions = []
                    for idx, scores in enumerate(all_scores):
                        sorted_scores = sorted(scores, key=lambda item: item["score"], reverse=True)
                        top = sorted_scores[0]
                        predictions.append(
                            {
                                "text": texts[idx][:120].replace("\\n", " "),
                                "true_label": label_map[labels[idx]],
                                "pred_label": top["label"],
                                "pred_score": round(top["score"], 4),
                                "neg_prob": round(sorted_scores[0]["score"] if sorted_scores[0]["label"] == "NEGATIVE" else sorted_scores[1]["score"], 4),
                                "pos_prob": round(sorted_scores[0]["score"] if sorted_scores[0]["label"] == "POSITIVE" else sorted_scores[1]["score"], 4),
                            }
                        )

                    df = pd.DataFrame(predictions)
                    display(df)

                    roc_auc = load_metric("roc_auc")
                    f1_metric = load_metric("f1")
                    preds = [0 if row["pred_label"] == "NEGATIVE" else 1 for row in predictions]
                    roc_score = roc_auc.compute(
                        references=labels,
                        prediction_scores=[row["pos_prob"] for row in predictions],
                    )["roc_auc"]
                    f1_score = f1_metric.compute(predictions=preds, references=labels)["f1"]

                    print(f"ROC-AUC: {roc_score:.3f} | F1: {f1_score:.3f}")

                    predictions_path = OUTPUT_DIR / "predictions.csv"
                    df.to_csv(predictions_path, index=False)
                    print(f"Saved predictions to {predictions_path}")
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        outputs = classifier(texts, batch_size=BATCH_SIZE, truncation=True, padding=True)
                        if outputs:
                            recorder.mark_first_token()
                        recorder.add_items(len(outputs))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="sentiment-imdb",
                        model_id=MODEL_ID,
                        dataset=DATASET_ID,
                        sequence_or_image_res="256-tokens",
                        batch=str(BATCH_SIZE),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/nlp/sentiment-distilbert-imdb_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/serving/serving-fastapi-pipeline_demo_cpu-first.ipynb"),
        title="Serving Demo",
        task_subtitle="FastAPI + Transformers pipeline",
        tldr="Expose a sentiment pipeline via FastAPI, test with an in-notebook client, and benchmark the core inference.",
        model_link="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english",
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        model_license="Apache-2.0",
        dataset_link="https://huggingface.co/datasets/imdb",
        dataset_name="IMDB samples",
        dataset_license="CC BY-NC 4.0",
        gotchas_link="../fixes-and-tips/fastapi-local-run.md",
        gotchas_label="Use FastAPI TestClient inside notebooks to avoid event loop clashes",
        colab_path="notebooks/serving/serving-fastapi-pipeline_demo_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Load the sentiment pipeline and initialise a FastAPI app.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path
                    from typing import Dict

                    import torch
                    from datasets import load_dataset
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    try:
                        from fastapi import FastAPI
                        from pydantic import BaseModel
                        from fastapi.testclient import TestClient
                    except ImportError as exc:  # noqa: BLE001
                        raise RuntimeError("Install fastapi and pydantic to run this notebook") from exc

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
                    OUTPUT_DIR = Path("outputs") / "fastapi-serving"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    sample_texts = load_dataset("imdb", split="test[:4]")["text"]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Build the API",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    classifier = pipeline(
                        "text-classification",
                        model=MODEL_ID,
                        device=DEVICE,
                        top_k=1,
                    )
                    load_time = time.perf_counter() - load_start

                    app = FastAPI(title="Sentiment Demo")

                    class Request(BaseModel):
                        text: str

                    @app.post("/predict")
                    def predict(request: Request) -> Dict[str, str]:
                        pred = classifier(request.text)[0]
                        return {"label": pred["label"], "score": f"{pred['score']:.4f}"}

                    client = TestClient(app)

                    payload = {"text": "I love CPU-friendly ML demos."}
                    response = client.post("/predict", json=payload)
                    response.json()
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        for idx, text in enumerate(sample_texts):
                            _ = classifier(text)
                            if idx == 0:
                                recorder.mark_first_token()
                            recorder.add_items(1)

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="fastapi-serving",
                        model_id=MODEL_ID,
                        dataset="imdb",
                        sequence_or_image_res="256-tokens",
                        batch="1",
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/serving/serving-fastapi-pipeline_demo_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/multimodal/retrieval-clip-mini-batch_effects_cpu-first.ipynb"),
        title="Batch Effects Study",
        task_subtitle="CLIP retrieval throughput",
        tldr="Measure how batch size impacts CLIP text embedding throughput on CPU/Metal/CUDA.",
        model_link="https://huggingface.co/openai/clip-vit-base-patch32",
        model_name="openai/clip-vit-base-patch32",
        model_license="MIT",
        dataset_link="https://huggingface.co/datasets/friends_quotes",
        dataset_name="Synthetic prompts (Friends quotes)",
        dataset_license="CC BY-NC-SA 4.0",
        gotchas_link="../fixes-and-tips/metal-backend-fallback.md",
        gotchas_label="Metal fallback covers cases without MPS acceleration",
        colab_path="notebooks/multimodal/retrieval-clip-mini-batch_effects_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Prepare CLIP and seed text prompts for throughput measurement.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import numpy as np
                    import pandas as pd
                    import torch
                    from datasets import load_dataset
                    from transformers import CLIPModel, CLIPTokenizer

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> torch.device:
                        if preference == "cuda" and torch.cuda.is_available():
                            return torch.device("cuda")
                        if preference == "mps" and torch.backends.mps.is_available():
                            return torch.device("mps")
                        return torch.device("cpu")

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "openai/clip-vit-base-patch32"
                    OUTPUT_DIR = Path("outputs") / "clip-batch-study"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    prompts = load_dataset("friends_quotes", split="train[:32]")["quote"]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Throughput sweep",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
                    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
                    load_time = time.perf_counter() - load_start

                    def embed_text(batch):
                        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(DEVICE)
                        with torch.inference_mode():
                            features = model.get_text_features(**inputs)
                        return features

                    batch_sizes = [1, 2, 4, 8, 16]
                    results = []

                    for batch in batch_sizes:
                        slices = [prompts[i : i + batch] for i in range(0, len(prompts), batch)]

                        def run_batch(recorder):
                            for idx, chunk in enumerate(slices):
                                outputs = embed_text(chunk)
                                if idx == 0:
                                    recorder.mark_first_token()
                                recorder.add_items(outputs.shape[0])

                        metrics = measure_memory_speed(run_batch)
                        results.append(
                            {
                                "batch_size": batch,
                                "ttfb_s": metrics.get("ttfb_s"),
                                "items_per_s": metrics.get("throughput_per_s"),
                                "peak_ram_mb": metrics.get("peak_ram_mb"),
                            }
                        )

                    df = pd.DataFrame(results)
                    display(df)
                    df.to_csv(OUTPUT_DIR / "batch_effects.csv", index=False)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Benchmark append",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    for row in results:
                        append_benchmark_row(
                            task="clip-batch-effects",
                            model_id=MODEL_ID,
                            dataset="friends_quotes",
                            sequence_or_image_res="text-avg-32",
                            batch=str(row["batch_size"]),
                            peak_ram_mb=fmt(row["peak_ram_mb"], 2),
                            peak_vram_mb="",
                            load_time_s=fmt(load_time, 2),
                            ttfb_s=fmt(row["ttfb_s"], 3),
                            tokens_per_s_or_images_per_s=fmt(row["items_per_s"], 3),
                            precision=PRECISION,
                            notebook_path="notebooks/multimodal/retrieval-clip-mini-batch_effects_cpu-first.ipynb",
                            repo_commit=repo_commit,
                        )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(results, fp, indent=2)
                    results
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/multimodal/captioning-blip-base_flickr8k_cpu-first.ipynb"),
        title="Image Captioning",
        task_subtitle="BLIP Base on Flickr8k",
        tldr="Generate captions for Flickr8k samples with BLIP and compute a toy BLEU score.",
        model_link="https://huggingface.co/Salesforce/blip-image-captioning-base",
        model_name="Salesforce/blip-image-captioning-base",
        model_license="BSD-3-Clause",
        dataset_link="https://huggingface.co/datasets/flickr8k",
        dataset_name="Flickr8k (test subset)",
        dataset_license="Creative Commons",
        gotchas_link="../fixes-and-tips/tokenizer-context-oom.md",
        gotchas_label="Long captions may need truncation to avoid tokenizer overflow",
        colab_path="notebooks/multimodal/captioning-blip-base_flickr8k_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Fetch a few Flickr8k images and prepare BLIP for captioning.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import numpy as np
                    import pandas as pd
                    import torch
                    from datasets import load_dataset
                    from evaluate import load as load_metric
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "Salesforce/blip-image-captioning-base"
                    OUTPUT_DIR = Path("outputs") / "blip-captioning"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    dataset = load_dataset("flickr8k", split="test[:4]")
                    images = dataset["image"]
                    references = [ann[0]["raw"] for ann in dataset["annotations"]]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Generate captions",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    captioner = pipeline(
                        "image-to-text",
                        model=MODEL_ID,
                        device=DEVICE,
                        max_new_tokens=32,
                    )
                    load_time = time.perf_counter() - load_start

                    captions = [captioner(image)[0]["generated_text"] for image in images]

                    df = pd.DataFrame(
                        {
                            "caption": captions,
                            "reference": references,
                        }
                    )
                    display(df)

                    bleu = load_metric("bleu")
                    tokenised_preds = [caption.split() for caption in captions]
                    tokenised_refs = [[ref.split()] for ref in references]
                    bleu_score = bleu.compute(predictions=tokenised_preds, references=tokenised_refs)["bleu"]
                    print(f"Toy BLEU: {bleu_score:.3f}")

                    df.to_csv(OUTPUT_DIR / "captions.csv", index=False)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        outputs = [captioner(image)[0]["generated_text"] for image in images]
                        if outputs:
                            recorder.mark_first_token()
                        recorder.add_items(sum(len(out.split()) for out in outputs))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="image-captioning",
                        model_id=MODEL_ID,
                        dataset="flickr8k",
                        sequence_or_image_res="varied",
                        batch=str(len(images)),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/multimodal/captioning-blip-base_flickr8k_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump({"bleu": bleu_score, **metrics}, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/audio/audio-classification-hubert-superb_cpu-first.ipynb"),
        title="Audio Classification",
        task_subtitle="HuBERT SUPERB keyword spotting",
        tldr="Run HuBERT-based audio classification on a tiny SUPERB slice with CPU defaults.",
        model_link="https://huggingface.co/superb/hubert-base-superb-ks",
        model_name="superb/hubert-base-superb-ks",
        model_license="Apache-2.0",
        dataset_link="https://huggingface.co/datasets/superb",
        dataset_name="SUPERB Keyword Spotting (subset)",
        dataset_license="Apache-2.0",
        gotchas_link="../fixes-and-tips/input-tokenizer-oom.md",
        gotchas_label="Trim audio or resample to avoid tokenizer RAM spikes",
        colab_path="notebooks/audio/audio-classification-hubert-superb_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Prepare HuBERT for keyword spotting on a toy SUPERB slice.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import pandas as pd
                    import torch
                    from datasets import load_dataset, Audio
                    from evaluate import load as load_metric
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "superb/hubert-base-superb-ks"
                    OUTPUT_DIR = Path("outputs") / "hubert-ks"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    dataset = load_dataset("superb", "ks", split="validation[:8]")
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
                    id2label = dataset.features["label"].int2str
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Classify keywords",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    classifier = pipeline(
                        "audio-classification",
                        model=MODEL_ID,
                        device=DEVICE,
                        top_k=1,
                    )
                    load_time = time.perf_counter() - load_start

                    predictions = []
                    for row in dataset:
                        audio = row["audio"]
                        pred = classifier({"array": audio["array"], "sampling_rate": audio["sampling_rate"]})[0]
                        predictions.append(
                            {
                                "label": pred["label"],
                                "score": pred["score"],
                                "true_label": id2label(row["label"]),
                            }
                        )

                    df = pd.DataFrame(predictions)
                    accuracy = (df["label"] == df["true_label"]).mean()
                    print(f"Accuracy: {accuracy:.3f}")
                    display(df)

                    df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    audio_batches = [
                        {"array": row["audio"]["array"], "sampling_rate": row["audio"]["sampling_rate"]}
                        for row in dataset
                    ]

                    def run_inference(recorder):
                        outputs = classifier(audio_batches)
                        if outputs:
                            recorder.mark_first_token()
                        recorder.add_items(len(outputs))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="audio-classification-ks",
                        model_id=MODEL_ID,
                        dataset="superb:ks",
                        sequence_or_image_res="1s@16kHz",
                        batch=str(len(audio_batches)),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/audio/audio-classification-hubert-superb_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/audio/asr-whisper-tiny-base_cpu-first.ipynb"),
        title="Speech-to-Text",
        task_subtitle="Whisper Tiny ASR",
        tldr="Transcribe short audio clips with Whisper Tiny on CPU and compare with references.",
        model_link="https://huggingface.co/openai/whisper-tiny",
        model_name="openai/whisper-tiny",
        model_license="MIT",
        dataset_link="https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy",
        dataset_name="LibriSpeech Dummy (toy)",
        dataset_license="CC BY 4.0",
        gotchas_link="../fixes-and-tips/colab-cuda-mismatch.md",
        gotchas_label="Colab CUDA wheels may mismatch; pin torch/whisper versions",
        colab_path="notebooks/audio/asr-whisper-tiny-base_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Load a dummy LibriSpeech slice and prepare the Whisper pipeline.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import numpy as np
                    import torch
                    from datasets import load_dataset, Audio
                    from evaluate import load as load_metric
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "openai/whisper-tiny"
                    OUTPUT_DIR = Path("outputs") / "whisper-tiny"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    dataset = load_dataset(
                        "hf-internal-testing/librispeech_asr_dummy",
                        "clean",
                        split="validation[:4]",
                    )
                    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Transcribe",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    asr = pipeline(
                        "automatic-speech-recognition",
                        model=MODEL_ID,
                        device=DEVICE,
                    )
                    load_time = time.perf_counter() - load_start

                    transcripts = []
                    for row in dataset:
                        audio = row["audio"]
                        pred = asr({"array": audio["array"], "sampling_rate": audio["sampling_rate"]})
                        transcripts.append(
                            {
                                "id": row["id"],
                                "reference": row["text"],
                                "prediction": pred["text"],
                            }
                        )

                    wer_metric = load_metric("wer")
                    wer = wer_metric.compute(
                        predictions=[row["prediction"].lower() for row in transcripts],
                        references=[row["reference"].lower() for row in transcripts],
                    )
                    print(f"WER: {wer:.3f}")

                    with open(OUTPUT_DIR / "transcripts.json", "w", encoding="utf-8") as fp:
                        json.dump(transcripts, fp, indent=2)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    audio_batches = [
                        {"array": row["audio"]["array"], "sampling_rate": row["audio"]["sampling_rate"]}
                        for row in dataset
                    ]

                    def run_inference(recorder):
                        for idx, audio in enumerate(audio_batches):
                            result = asr(audio)
                            if idx == 0:
                                recorder.mark_first_token()
                            recorder.add_items(len(result["text"].split()))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="whisper-asr",
                        model_id=MODEL_ID,
                        dataset="hf-internal-testing/librispeech_asr_dummy",
                        sequence_or_image_res="15s@16kHz",
                        batch=str(len(audio_batches)),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/audio/asr-whisper-tiny-base_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump({"wer": wer, **metrics}, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/vision/detection-detr-resnet50_cpu-first.ipynb"),
        title="Object Detection",
        task_subtitle="DETR ResNet-50 quickstart",
        tldr="Detect objects on sample images with DETR and export annotated overlays.",
        model_link="https://huggingface.co/facebook/detr-resnet-50",
        model_name="facebook/detr-resnet-50",
        model_license="Apache-2.0",
        dataset_link="https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils",
        dataset_name="Fixture images (toy)",
        dataset_license="CC BY 4.0",
        gotchas_link="../fixes-and-tips/torch-compile-mps-quirks.md",
        gotchas_label="Skip torch.compile on MPS to avoid kernel mismatch",
        colab_path="notebooks/vision/detection-detr-resnet50_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Load sample images and prepare the DETR pipeline.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import numpy as np
                    import torch
                    from datasets import load_dataset
                    from PIL import Image, ImageDraw
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")
                    SCORE_THRESHOLD = float(os.environ.get("HF_SCORE_THRESHOLD", "0.7"))

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "facebook/detr-resnet-50"
                    OUTPUT_DIR = Path("outputs") / "detr"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    ds = load_dataset("hf-internal-testing/fixtures_image_utils", split="test[:3]")
                    images = [Image.fromarray(example["image"]) for example in ds]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Detect objects",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    detector = pipeline(
                        "object-detection",
                        model=MODEL_ID,
                        device=DEVICE,
                        threshold=SCORE_THRESHOLD,
                    )
                    load_time = time.perf_counter() - load_start

                    detections = detector(images)

                    expected_labels = [
                        {"image": 0, "labels": {"remote", "cat", "tv"}},
                        {"image": 1, "labels": {"remote", "book"}},
                        {"image": 2, "labels": {"dog", "person", "car"}},
                    ]
                    label_lookup = {entry["image"]: entry["labels"] for entry in expected_labels}

                    toy_hits = []
                    annotated_paths = []
                    for idx, (img, preds) in enumerate(zip(images, detections)):
                        draw = ImageDraw.Draw(img)
                        matched = 0
                        for pred in preds:
                            box = pred["box"]
                            label = pred["label"]
                            score = pred["score"]
                            if score < SCORE_THRESHOLD:
                                continue
                            draw.rectangle(
                                [(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])],
                                outline="lime",
                                width=3,
                            )
                            draw.text((box["xmin"], box["ymin"] - 10), f"{label} {score:.2f}", fill="lime")
                            if label in label_lookup.get(idx, set()):
                                matched += 1
                        toy_hits.append(matched / max(1, len(label_lookup.get(idx, set()))))
                        out_path = OUTPUT_DIR / f"annotated_{idx}.png"
                        img.save(out_path)
                        annotated_paths.append(out_path)

                    toy_map = float(np.mean(toy_hits))
                    print(f"Toy mAP (label hit ratio): {toy_map:.3f}")
                    print("Annotated images:", annotated_paths)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        results = detector(images)
                        if results:
                            recorder.mark_first_token()
                        recorder.add_items(sum(len(r) for r in results))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="detr-detection",
                        model_id=MODEL_ID,
                        dataset="hf-internal-testing/fixtures_image_utils",
                        sequence_or_image_res="varied",
                        batch=str(len(images)),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/vision/detection-detr-resnet50_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump({"toy_map": toy_map, **metrics}, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/vision/zero-shot-clip-retrieval_cpu-first.ipynb"),
        title="Zero-shot Retrieval",
        task_subtitle="CLIP text-image matching",
        tldr="Embed images and text prompts with CLIP to rank matches on CPU, with Metal/CUDA toggles for acceleration.",
        model_link="https://huggingface.co/openai/clip-vit-base-patch32",
        model_name="openai/clip-vit-base-patch32",
        model_license="MIT",
        dataset_link="https://huggingface.co/datasets/nateraw/horses_or_humans",
        dataset_name="Horses or Humans (tiny sample)",
        dataset_license="Apache-2.0",
        gotchas_link="../fixes-and-tips/metal-backend-fallback.md",
        gotchas_label="Metal fallback ensures CLIP works even without MPS",
        colab_path="notebooks/vision/zero-shot-clip-retrieval_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Fetch a tiny set of images and prepare CLIP for retrieval.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import numpy as np
                    import pandas as pd
                    import torch
                    from datasets import load_dataset
                    from PIL import Image
                    from transformers import CLIPModel, CLIPProcessor

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> torch.device:
                        if preference == "cuda" and torch.cuda.is_available():
                            return torch.device("cuda")
                        if preference == "mps" and torch.backends.mps.is_available():
                            return torch.device("mps")
                        return torch.device("cpu")

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "openai/clip-vit-base-patch32"
                    OUTPUT_DIR = Path("outputs") / "clip-retrieval"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    dataset = load_dataset("nateraw/horses_or_humans", split="train[:4]")
                    images = [Image.fromarray(item["image"]) for item in dataset]
                    text_queries = [
                        "A person performing yoga outdoors",
                        "A horse jumping over a barrier",
                        "A close-up portrait of a human face",
                    ]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Embed & score",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    processor = CLIPProcessor.from_pretrained(MODEL_ID)
                    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
                    load_time = time.perf_counter() - load_start

                    def embed_images(batch):
                        inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
                        with torch.inference_mode():
                            features = model.get_image_features(**inputs)
                        return features / features.norm(dim=-1, keepdim=True)

                    def embed_text(batch):
                        inputs = processor(text=batch, return_tensors="pt", padding=True).to(DEVICE)
                        with torch.inference_mode():
                            features = model.get_text_features(**inputs)
                        return features / features.norm(dim=-1, keepdim=True)

                    image_embeddings = embed_images(images)
                    text_embeddings = embed_text(text_queries)

                    sims = (text_embeddings @ image_embeddings.T).cpu().numpy()
                    df = pd.DataFrame(
                        sims,
                        columns=[f"image_{idx}" for idx in range(len(images))],
                        index=text_queries,
                    )
                    display(df.style.format("{:.3f}"))

                    rankings = {
                        query: df.loc[query].nlargest(3).index.tolist()
                        for query in text_queries
                    }
                    print(json.dumps(rankings, indent=2))
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        img_feat = embed_images(images)
                        txt_feat = embed_text(text_queries)
                        recorder.mark_first_token()
                        recorder.add_items(len(text_queries) * len(images))
                        return img_feat, txt_feat

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="clip-retrieval",
                        model_id=MODEL_ID,
                        dataset="nateraw/horses_or_humans",
                        sequence_or_image_res="224x224",
                        batch=str(len(images)),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/vision/zero-shot-clip-retrieval_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/vision/classification-vit-base-224-imagenette_cpu-first.ipynb"),
        title="Image Classification",
        task_subtitle="ViT Base on Imagenette",
        tldr="Compare zero-shot ViT predictions on Imagenette samples with CPU-first defaults.",
        model_link="https://huggingface.co/google/vit-base-patch16-224",
        model_name="google/vit-base-patch16-224",
        model_license="Apache-2.0",
        dataset_link="https://huggingface.co/datasets/frgfm/imagenette",
        dataset_name="Imagenette (validation split)",
        dataset_license="Apache-2.0",
        gotchas_link="../fixes-and-tips/torch-compile-mps-quirks.md",
        gotchas_label="Disable torch.compile on MPS until PyTorch resolves precision drift",
        colab_path="notebooks/vision/classification-vit-base-224-imagenette_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Load a small Imagenette validation slice and prepare the ViT classifier.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from collections import Counter
                    from pathlib import Path

                    import pandas as pd
                    import torch
                    from datasets import load_dataset
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")
                    BATCH_SIZE = int(os.environ.get("HF_BATCH", "4"))

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    MODEL_ID = "google/vit-base-patch16-224"
                    DATASET_ID = "frgfm/imagenette"
                    SPLIT = "validation[:16]"
                    OUTPUT_DIR = Path("outputs") / "vit-imagenette"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    ds = load_dataset(DATASET_ID, split=SPLIT)
                    id2label = ds.features["label"].int2str
                    images = ds["image"]
                    labels = ds["label"]
                    true_labels = [id2label(idx) for idx in labels]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Classify",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    classifier = pipeline(
                        "image-classification",
                        model=MODEL_ID,
                        device=DEVICE,
                        top_k=1,
                    )
                    load_time = time.perf_counter() - load_start

                    predictions = classifier(images)
                    pred_labels = [pred[0]["label"] if isinstance(pred, list) else pred["label"] for pred in predictions]

                    accuracy = sum(pl == tl for pl, tl in zip(pred_labels, true_labels)) / len(true_labels)
                    print(f"Accuracy on {len(true_labels)} samples: {accuracy:.3f}")

                    confusion = (
                        pd.crosstab(pd.Series(true_labels, name="true"), pd.Series(pred_labels, name="pred"))
                        .reindex(index=sorted(set(true_labels)), columns=sorted(set(pred_labels)))
                        .fillna(0)
                    )
                    display(confusion)

                    results = pd.DataFrame(
                        {
                            "true_label": true_labels,
                            "pred_label": pred_labels,
                            "score": [pred[0]["score"] if isinstance(pred, list) else pred["score"] for pred in predictions],
                        }
                    )
                    results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        outputs = classifier(images)
                        if outputs:
                            recorder.mark_first_token()
                        recorder.add_items(len(outputs))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="imagenette-classification",
                        model_id=MODEL_ID,
                        dataset=DATASET_ID,
                        sequence_or_image_res="224x224",
                        batch=str(BATCH_SIZE),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/vision/classification-vit-base-224-imagenette_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/nlp/instruct-generation-llama-3-instruct-8b_lite_cpu-first.ipynb"),
        title="Instruction Generation",
        task_subtitle="Llama 3 Instruct 8B (lite fallback)",
        tldr="Demonstrate small instruction prompts on CPU with a TinyLlama fallback while documenting access needs for Llama 3.",
        model_link="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
        model_name="Meta-Llama-3-8B-Instruct (fallback: TinyLlama-1.1B-Chat)",
        model_license="Custom (Llama 3), Apache-2.0 (TinyLlama)",
        dataset_link="https://huggingface.co/datasets/prompt-injection/ultra-chat",
        dataset_name="Synthetic prompts (UltraChat sample)",
        dataset_license="CC BY-SA 4.0",
        gotchas_link="../fixes-and-tips/bitsandbytes-wheel-mismatch.md",
        gotchas_label="bitsandbytes optional; CPU uses fp32 pipeline automatically",
        colab_path="notebooks/nlp/instruct-generation-llama-3-instruct-8b_lite_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Load the instruct model with a CPU-safe fallback if the full Llama 3 weights are inaccessible.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import torch
                    from transformers import AutoTokenizer, pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    TARGET_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
                    FALLBACK_MODEL = os.environ.get("HF_FALLBACK_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                    REQUESTED_MODEL = os.environ.get("HF_MODEL_ID", FALLBACK_MODEL)

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    def choose_model() -> str:
                        target = TARGET_MODEL if REQUESTED_MODEL == TARGET_MODEL else REQUESTED_MODEL
                        if target == TARGET_MODEL:
                            try:
                                AutoTokenizer.from_pretrained(TARGET_MODEL, token=os.environ.get("HF_TOKEN"))
                                print("Using Meta-Llama-3-8B-Instruct (ensure you accepted the license).")
                                return TARGET_MODEL
                            except Exception as error:  # noqa: BLE001
                                print(f"Falling back to {FALLBACK_MODEL} because Llama 3 load failed: {error}")
                        return FALLBACK_MODEL

                    MODEL_ID = choose_model()
                    OUTPUT_DIR = Path("outputs") / "instruct-generation"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    prompts = [
                        "Summarise the repo's CPU-first philosophy in one sentence.",
                        "Give me two bullet tips for speeding up Transformers inference on CPUs.",
                        "When should I enable 4-bit quantization? Respond in 3 short bullet points.",
                    ]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Generate responses",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    generation_kwargs = {
                        "temperature": 0.7,
                        "max_new_tokens": 128,
                        "do_sample": True,
                        "pad_token_id": None,
                    }

                    torch.manual_seed(42)
                    load_start = time.perf_counter()
                    generator = pipeline(
                        "text-generation",
                        model=MODEL_ID,
                        device=DEVICE,
                        torch_dtype=torch.float32,
                    )
                    load_time = time.perf_counter() - load_start

                    outputs = []
                    for prompt in prompts:
                        response = generator(prompt, **generation_kwargs)[0]["generated_text"]
                        outputs.append({"prompt": prompt, "response": response})
                        print("\\n---\\n")
                        print(response)

                    with open(OUTPUT_DIR / "generations.json", "w", encoding="utf-8") as fp:
                        json.dump(outputs, fp, indent=2)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        for idx, prompt in enumerate(prompts):
                            result = generator(prompt, **generation_kwargs)
                            if idx == 0:
                                recorder.mark_first_token()
                            recorder.add_items(len(result[0]["generated_text"].split()))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="instruction-generation",
                        model_id=MODEL_ID,
                        dataset="synthetic-prompts",
                        sequence_or_image_res="128-tokens",
                        batch="1",
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/nlp/instruct-generation-llama-3-instruct-8b_lite_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/nlp/summarization-t5-small-cnn_dm_cpu-first.ipynb"),
        title="Summarisation",
        task_subtitle="T5-small on CNN/DailyMail",
        tldr="Summarise news articles with T5-small and validate ROUGE on a tiny CPU slice.",
        model_link="https://huggingface.co/t5-small",
        model_name="t5-small",
        model_license="Apache-2.0",
        dataset_link="https://huggingface.co/datasets/cnn_dailymail",
        dataset_name="CNN/DailyMail 3.0.0",
        dataset_license="NonCommercial",
        gotchas_link="../fixes-and-tips/tokenizer-context-oom.md",
        gotchas_label="Long contexts can OOM tokenizers—clip inputs to 512 tokens",
        colab_path="notebooks/nlp/summarization-t5-small-cnn_dm_cpu-first.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Slice the CNN/DailyMail dataset and prepare the summarisation pipeline.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import pandas as pd
                    import torch
                    from datasets import load_dataset
                    from evaluate import load as load_metric
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")
                    MAX_INPUT_TOKENS = int(os.environ.get("HF_CONTEXT_LEN", "512"))
                    BATCH_SIZE = int(os.environ.get("HF_BATCH", "2"))

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}, max_input_tokens={MAX_INPUT_TOKENS}")

                    MODEL_ID = "t5-small"
                    DATASET_ID = "cnn_dailymail"
                    DATASET_CONFIG = "3.0.0"
                    OUTPUT_DIR = Path("outputs") / "summarization-t5-small"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="test[:4]")
                    sources = dataset["article"]
                    references = dataset["highlights"]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Generate summaries",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    summariser = pipeline(
                        "summarization",
                        model=MODEL_ID,
                        device=DEVICE,
                        batch_size=BATCH_SIZE,
                        truncation=True,
                        max_length=128,
                        min_length=32,
                    )
                    load_time = time.perf_counter() - load_start

                    generated = summariser(
                        sources,
                        truncation=True,
                        max_length=128,
                        min_length=32,
                        clean_up_tokenization_spaces=True,
                    )
                    summaries = [item["summary_text"] for item in generated]

                    comparison = pd.DataFrame(
                        {
                            "source": [src[:200].replace("\\n", " ") + "…" for src in sources],
                            "summary": summaries,
                            "reference": references,
                        }
                    )
                    display(comparison)

                    rouge = load_metric("rouge")
                    rouge_scores = rouge.compute(predictions=summaries, references=references, use_stemmer=True)
                    print({k: round(v.mid.fmeasure, 3) for k, v in rouge_scores.items()})

                    out_path = OUTPUT_DIR / "summaries.csv"
                    comparison.to_csv(out_path, index=False)
                    print(f"Wrote {out_path}")
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        outputs = summariser(
                            sources,
                            truncation=True,
                            max_length=128,
                            min_length=32,
                        )
                        if outputs:
                            recorder.mark_first_token()
                        recorder.add_items(len(outputs))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="summarization-cnn_dm",
                        model_id=MODEL_ID,
                        dataset=f"{DATASET_ID}:{DATASET_CONFIG}",
                        sequence_or_image_res=f"{MAX_INPUT_TOKENS}-tokens",
                        batch=str(BATCH_SIZE),
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/nlp/summarization-t5-small-cnn_dm_cpu-first.ipynb",
                        repo_commit=repo_commit,
                    )

                    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as fp:
                        json.dump(metrics, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
    NotebookSpec(
        path=Path("notebooks/serving/tgi-vs-pipeline-latency_microbenchmark.ipynb"),
        title="Latency Microbenchmark",
        task_subtitle="TGI vs. Transformers pipeline (concept)",
        tldr="Capture baseline pipeline latency on CPU and scaffold fields for future Text Generation Inference runs.",
        model_link="https://huggingface.co/docs/text-generation-inference/index",
        model_name="Text Generation Inference (pending)",
        model_license="Apache-2.0",
        dataset_link="https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
        dataset_name="UltraChat prompts (sample)",
        dataset_license="CC BY-SA 4.0",
        gotchas_link="../fixes-and-tips/tgi-setup-todo.md",
        gotchas_label="TGI requires container and GPU setup—tracked in Fixes entry",
        colab_path="notebooks/serving/tgi-vs-pipeline-latency_microbenchmark.ipynb",
        body_cells=[
            {
                "type": "markdown",
                "source": textwrap.dedent(
                    """
                    ## Setup
                    Measure pipeline latency now and leave TODO hooks for a future TGI deployment.
                    """
                ),
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    import json
                    import os
                    import subprocess
                    import time
                    from pathlib import Path

                    import torch
                    from datasets import load_dataset
                    from transformers import pipeline

                    from notebooks._templates.measure import append_benchmark_row, measure_memory_speed

                    DEVICE_PREFERENCE = os.environ.get("HF_DEVICE", "cpu")
                    PRECISION = os.environ.get("HF_PRECISION", "fp32")

                    def resolve_device(preference: str = "cpu") -> str:
                        if preference == "cuda" and torch.cuda.is_available():
                            return "cuda:0"
                        if preference == "mps" and torch.backends.mps.is_available():
                            return "mps"
                        return "cpu"

                    DEVICE = resolve_device(DEVICE_PREFERENCE)
                    print(f"Using device={DEVICE}")

                    PIPELINE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    OUTPUT_DIR = Path("outputs") / "tgi-vs-pipeline"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                    prompts = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:4]")["prompt"]
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Pipeline latency (baseline)",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    torch.manual_seed(42)

                    load_start = time.perf_counter()
                    generator = pipeline(
                        "text-generation",
                        model=PIPELINE_MODEL_ID,
                        device=DEVICE,
                        max_new_tokens=64,
                        do_sample=False,
                    )
                    load_time = time.perf_counter() - load_start

                    pipeline_outputs = []
                    for prompt in prompts:
                        pipeline_outputs.append(generator(prompt)[0]["generated_text"])

                    with open(OUTPUT_DIR / "pipeline_outputs.json", "w", encoding="utf-8") as fp:
                        json.dump(pipeline_outputs, fp, indent=2)
                    """
                ),
            },
            {
                "type": "markdown",
                "source": "## Measurement",
            },
            {
                "type": "code",
                "source": textwrap.dedent(
                    """
                    def run_inference(recorder):
                        for idx, prompt in enumerate(prompts):
                            result = generator(prompt, max_new_tokens=64, do_sample=False)
                            if idx == 0:
                                recorder.mark_first_token()
                            recorder.add_items(len(result[0]["generated_text"].split()))

                    metrics = measure_memory_speed(run_inference)

                    def fmt(value, digits=4):
                        if value in (None, "", float("inf")):
                            return ""
                        return f"{value:.{digits}f}"

                    try:
                        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                    except Exception:  # noqa: BLE001
                        repo_commit = ""

                    append_benchmark_row(
                        task="tgi-pipeline-baseline",
                        model_id=PIPELINE_MODEL_ID,
                        dataset="ultrachat_200k",
                        sequence_or_image_res="64-tokens",
                        batch="1",
                        peak_ram_mb=fmt(metrics.get("peak_ram_mb"), 2),
                        peak_vram_mb=fmt(metrics.get("peak_vram_mb"), 2),
                        load_time_s=fmt(load_time, 2),
                        ttfb_s=fmt(metrics.get("ttfb_s"), 3),
                        tokens_per_s_or_images_per_s=fmt(metrics.get("throughput_per_s"), 3),
                        precision=PRECISION,
                        notebook_path="notebooks/serving/tgi-vs-pipeline-latency_microbenchmark.ipynb",
                        repo_commit=repo_commit,
                    )

                    TODO_TGI_NOTES = {
                        "status": "pending",
                        "notes": "Provision Text Generation Inference container and populate compare.csv",
                    }
                    with open(OUTPUT_DIR / "tgi_todo.json", "w", encoding="utf-8") as fp:
                        json.dump({"metrics": metrics, "tgi": TODO_TGI_NOTES}, fp, indent=2)
                    metrics
                    """
                ),
            },
        ],
    ),
]


def top_cell(spec: NotebookSpec) -> str:
    lines = [
        f"# {spec['title']} — {spec['task_subtitle']}",
        f"**TL;DR:** {spec['tldr']}",
        "",
        f"**Models & Datasets:** [{spec['model_name']}]({spec['model_link']}) ({spec['model_license']}), [{spec['dataset_name']}]({spec['dataset_link']}) ({spec['dataset_license']})",
        f"**Run Profiles:** {RUN_PROFILES}",
        f"**Env (minimal):** {BASE_ENV}",
        f"**Colab:** [Open in Colab](https://colab.research.google.com/github/SSusantAchary/Hands-On-Huggingface-AI-Models/blob/main/{spec['colab_path']})",
        "",
        "**Switches (edit in one place):**",
        SWITCHES,
        "",
        "**Footprint & Speed (fill after run):**",
        "- Peak RAM: TODO",
        "- Peak VRAM: TODO (if GPU)",
        "- TTFB: TODO, Throughput: TODO, Load time: TODO",
        "",
        f"**Gotchas:** {spec['gotchas_label']} ([Fixes & Tips]({spec['gotchas_link']}))",
    ]
    return "\n".join(lines)


def closing_cell(notebook_path: str) -> str:
    return textwrap.dedent(
        f"""
        ## Results Summary
        - Observations: TODO
        - Metrics captured: see `benchmarks/matrix.csv`

        ## Next Steps
        - TODOs: fill in after benchmarking

        ## Repro
        - Seed: 42 (set in measurement cell)
        - Libraries: captured via `detect_env()`
        - Notebook path: `{notebook_path}`
        - Latest commit: populated automatically when appending benchmarks (if git available)
        """.strip()
    )


def write_notebook(spec: NotebookSpec) -> None:
    def to_source(text: str) -> List[str]:
        lines = text.rstrip().splitlines()
        return [line + "\n" for line in lines]

    cells = []
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": to_source(top_cell(spec)),
        }
    )
    for cell in spec["body_cells"]:
        if cell.get("type") == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": to_source(cell["source"])})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": to_source(cell["source"]),
                }
            )
    cells.append({"cell_type": "markdown", "metadata": {}, "source": to_source(closing_cell(spec["colab_path"]))})

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = REPO / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(nb, fp, indent=2)
    print("Wrote", path)


def main() -> None:
    for spec in NOTEBOOKS:
        write_notebook(spec)


if __name__ == "__main__":
    main()
