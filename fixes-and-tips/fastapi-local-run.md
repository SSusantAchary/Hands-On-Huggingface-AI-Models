# FastAPI local run conflict
**Symptom:** `RuntimeError: This event loop is already running` when launching Uvicorn inside a notebook.
**Root cause:** Notebooks already run an asyncio event loop, so spawning `uvicorn.run` directly blocks.
**Fix:** Use `fastapi.testclient.TestClient` or launch Uvicorn via a terminal (`uvicorn app:app --reload`).
**Verify:** Calling the `/predict` endpoint through `TestClient` returns JSON without raising an exception.
**Scope:** fastapi>=0.110, uvicorn>=0.29, notebooks
**Related:** notebooks/serving/serving-fastapi-pipeline_demo_cpu-first.ipynb
