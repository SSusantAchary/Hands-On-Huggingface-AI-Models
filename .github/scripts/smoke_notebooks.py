"""Execute the first cell of every notebook to spot import regressions early."""
from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def iter_notebooks() -> list[Path]:
    notebooks = []
    for path in NOTEBOOK_DIR.rglob("*.ipynb"):
        if "_templates" in path.parts:
            continue
        notebooks.append(path)
    return sorted(notebooks)


def smoke(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    if not nb.cells:
        return
    smoke_nb = nbformat.v4.new_notebook(metadata=nb.metadata)
    smoke_nb.cells = [nb.cells[0]]
    client = NotebookClient(smoke_nb, timeout=120, kernel_name="python3")
    client.execute()


def main() -> None:
    failures = []
    for notebook in iter_notebooks():
        try:
            smoke(notebook)
            print(f"✅ {notebook.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {notebook.relative_to(ROOT)} :: {exc}")
            failures.append(notebook)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
