#!/usr/bin/env python3
"""
Generate placeholder notebooks for every entry in meta/notebook_catalog.csv.

Each notebook lives under notebooks/<section>/ and uses a consistent naming
scheme (section-index-slug.ipynb). Existing notebooks are left untouched.
The script also updates the notebook_path column in the catalog CSV when a
placeholder is created or detected.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "meta" / "notebook_catalog.csv"
NOTEBOOK_ROOT = ROOT / "notebooks"

KERNEL_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {
        "name": "python",
        "version": "3.10",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
    },
}


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "model"


def build_notebook(model_name: str, model_id: str, use_case: str) -> Dict:
    header = [
        f"# {model_name} — {use_case}\n",
        "**TL;DR:** TODO – summarize the task once the notebook is implemented.\n",
        "\n",
        f"**Models & Datasets:** [{model_name}](https://huggingface.co/{model_id}) (model license review required)\n",
        "**Run Profiles:** 🖥️ CPU | 🍎 Metal (Apple Silicon) | ⚡ CUDA GPU\n",
        "**Env (baseline):** python>=3.10, transformers, datasets, evaluate (add extras per task)\n",
        f"**Colab:** TODO — add badge when notebook is published\n",
        "\n",
        "**Switches (edit in one place):**\n",
        "- `device` = {\"cpu\",\"mps\",\"cuda\"}\n",
        "- `precision` = {\"fp32\",\"fp16\",\"bf16\",\"int8\",\"4bit\"}\n",
        "- Task-specific knobs (context length, image size, batch size)\n",
        "\n",
        "**Footprint & Speed (fill after run):**\n",
        "- Peak RAM: TODO\n",
        "- Peak VRAM: TODO (if GPU)\n",
        "- TTFB: TODO, Throughput: TODO\n",
        "\n",
        "**Gotchas:** Link to Fixes entry once discovered.\n",
    ]

    planning = [
        "## TODO Checklist\n",
        "- [ ] Outline dataset download / preparation steps\n",
        "- [ ] Implement inference + evaluation cells\n",
        "- [ ] Use `notebooks/_templates/measure.py` to log metrics and append to `benchmarks/matrix.csv`\n",
        "- [ ] Update `/meta/notebook_catalog.csv` with status and Colab link after publishing\n",
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": KERNEL_METADATA,
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": header},
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Placeholder imports — extend when implementing\n",
                    "from pathlib import Path\n",
                    "import os\n",
                    "import json\n",
                    "\n",
                    "# TODO: add task-specific libraries (e.g., transformers, datasets)\n",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": planning},
        ],
    }
    return notebook


def main() -> None:
    if not CATALOG_CSV.exists():
        raise SystemExit("Catalog CSV not found. Run build_catalog.py first.")

    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)

    with CATALOG_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    counts: Dict[str, int] = {}
    for row in rows:
        section = row["section"].strip().lower()
        if section not in {"audio", "multimodal", "nlp", "vision"}:
            raise SystemExit(f"Unknown section '{section}' in catalog.")
        counts.setdefault(section, 0)
        counts[section] += 1
        index = counts[section]

        slug = slugify(row["model_name"])
        subdir = Path(section) / f"{section}_notebooks"
        filename = f"{section}-{index:02d}-{slug}.ipynb"
        relative_path = subdir / filename
        notebook_path = NOTEBOOK_ROOT / relative_path

        existing_path_str = row["notebook_path"].strip()
        if existing_path_str and existing_path_str != "TODO":
            existing_path = NOTEBOOK_ROOT / existing_path_str
            if existing_path.exists() and existing_path.resolve() != notebook_path.resolve():
                notebook_path.parent.mkdir(parents=True, exist_ok=True)
                existing_path.rename(notebook_path)
        old_layout = NOTEBOOK_ROOT / Path(section) / filename
        if old_layout.exists() and old_layout.resolve() != notebook_path.resolve():
            notebook_path.parent.mkdir(parents=True, exist_ok=True)
            old_layout.rename(notebook_path)

        if not notebook_path.exists():
            notebook_path.parent.mkdir(parents=True, exist_ok=True)
            nb = build_notebook(row["model_name"], row["model_id"], row["use_case"])
            with notebook_path.open("w", encoding="utf-8") as handle:
                json.dump(nb, handle, indent=2)
            print(f"Created {notebook_path.relative_to(ROOT)}")
        else:
            print(f"Ensured {notebook_path.relative_to(ROOT)}")

        row["notebook_path"] = str(relative_path).replace("\\", "/")

    with CATALOG_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Catalog CSV updated with notebook paths.")


if __name__ == "__main__":
    main()
