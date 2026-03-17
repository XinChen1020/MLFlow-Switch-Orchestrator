# MLFlow Switch Orchestrator

MLFlow Switch Orchestrator is a lightweight ML deployment system for spec-driven training, MLflow-backed lineage, controlled model promotion, and zero-downtime serving behind a stable inference endpoint.

The main workflow is:

- launch training from a spec
- pre-create and track the run in MLflow
- register the resulting model version
- start a candidate serving container and wait for health checks
- switch traffic without changing the public inference endpoint

Key behaviors:

- **Spec-driven training**: trainer definitions live in `router/specs/spec.yaml`.
- **MLflow lineage**: the router injects `MLFLOW_RUN_ID` into the trainer container and resolves the model version created by that run.
- **Controlled promotion**: rollout happens only after a candidate serving container is healthy.
- **Stable serving endpoint**: the active model can change without changing the public inference URL.
- **Serving-only promotion**: existing model versions or aliases can be promoted through `/admin/roll`.
- **Explicit rollback**: the previous active deployment is recorded on each successful promotion and can be restored through `/admin/rollback`.

## System Overview

- **Router service (`router/`)**: FastAPI control plane that loads trainer specs, launches training containers, tracks MLflow runs, resolves model versions, and coordinates rollout.
- **Trainer containers (`model-images/`)**: model-specific training jobs that log parameters, metrics, datasets, and registered models to MLflow.
- **Serve containers**: runtime images that load a promoted MLflow model URI and expose inference APIs.
- **MLflow tracking server**: central source of truth for experiment runs, model versions, aliases, and artifacts.
- **Caddy reverse proxy**: stable public inference entrypoint that is flipped only after a candidate server is healthy.
- **Docker socket proxy**: narrows the router's Docker access surface while still allowing container lifecycle automation.

## Architecture Diagram

flowchart TD
    A[Trainer spec] --> R[Router control plane]
    B[Train / train_then_roll request] --> R

    R -->|pre-create run| M[MLflow tracking]
    R -->|launch trainer container| T[Sklearn or PyTorch trainer]
    T -->|log params, metrics, datasets, model| M
    M --> G[MLflow registry]

    R -->|resolve model version or alias| G
    R -->|start candidate serve container| S[Serve container]
    S -->|health check passes| P[Caddy stable endpoint]
    P --> C[Client /invocations]

    R --> X[Persist active + previous deployment]
    X --> RB[Rollback available]

## Reference Backends

- **`sklearn-model-1`**: a scikit-learn random forest regressor on the diabetes dataset.
- **`pytorch-model-1`**: a small PyTorch MLP regressor on the same dataset, logged through `mlflow.pytorch` and served through the same rollout path.

The two backends are intentionally similar so the main difference is the training framework rather than the deployment flow.

## Dataset Modes

The bundled backends default to the checked-in `demo_data/diabetes.csv` dataset,
which is copied into the trainer images at build time. This keeps the demo path
deterministic while still allowing custom data.

- Default demo mode: the trainer spec sets `DATASET_PATH=/app/demo_data/diabetes.csv` and `TARGET_COLUMN=target`.
- Custom dataset mode: override `DATASET_PATH` and `TARGET_COLUMN` in the train request parameters or spec.
- Fast demo mode: optionally set `DATASET_SAMPLE_ROWS` to train on a deterministic sample of the same dataset.
- Dataset metadata: `DATASET_NAME` and `DATASET_VERSION` are logged to MLflow alongside the run.

If you want to regenerate the checked-in demo CSV:

```bash
cd model-images/sklearn-model-1
uv run python ../../demo_data/prepare_demo_data.py
```

## End-to-End Flow

1. A request hits `/admin/train/{trainer}` or `/admin/train_then_roll/{trainer}`.
2. The router resolves the trainer spec and pre-creates an MLflow run.
3. The trainer container receives `MLFLOW_RUN_ID` plus any spec-defined or request-level parameters.
4. The training job logs dataset lineage, hyperparameters, evaluation metrics, and a registered model into MLflow.
5. The router resolves the model version associated with that exact run.
6. For rollout, the router starts a candidate serving container pointed at `models:/<name>/<version>`.
7. If the candidate passes health checks, the proxy flips traffic to it and the previous serving container is retired.
8. Clients continue using the same inference endpoint while the production model version changes underneath it.

## Training and Tracking

The reference trainer in `model-images/sklearn-model-1/` demonstrates the main ML workflow:

- dataset lineage is logged for training, validation, and evaluation splits
- hyperparameters and dataset metadata are recorded on the run
- evaluation uses `mlflow.evaluate` for standardized regression metrics
- the trained model is registered into the MLflow Model Registry
- the rollout path updates the production alias after a successful deployment

## Getting Started

1. Create a `.env` with any overrides for the variables referenced in `docker-compose.prod.yaml`. The defaults are enough for local testing.
2. Build the reference trainer and serving images used by the bundled `sklearn-model-1` spec:
   ```bash
   docker compose -f model-images/sklearn-model-1/docker-compose.prod.yml build
   ```
   To build the PyTorch reference backend instead:
   ```bash
   docker compose -f model-images/pytorch-model-1/docker-compose.prod.yml build
   ```
3. Launch the stack:
   ```bash
   docker compose -f docker-compose.prod.yaml up --build
   ```
4. Trigger a train-and-roll flow:
   ```bash
   curl -X POST \
     'http://localhost:8000/admin/train_then_roll/sklearn-model-1' \
     -H 'Content-Type: application/json' \
     -d '{"wait_seconds": 600, "parameters": {"DATASET_SAMPLE_ROWS": 96, "N_ESTIMATORS": 32}}'
   ```
   Or run the PyTorch backend:
   ```bash
   curl -X POST \
     'http://localhost:8000/admin/train_then_roll/pytorch-model-1' \
     -H 'Content-Type: application/json' \
     -d '{"wait_seconds": 600, "parameters": {"DATASET_SAMPLE_ROWS": 96, "EPOCHS": 40, "HIDDEN_DIM": 16}}'
   ```
5. Query the stable inference endpoint after rollout completes:
   ```bash
   curl -X POST \
     'http://localhost:9000/invocations' \
     -H 'Content-Type: application/json' \
     -d '{
       "dataframe_split": {
         "columns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
         "data": [[0.03, 1, 0.06, 0.03, 0.04, 0.03, 0.02, 0.03, 0.04, 0.01]]
       }
     }'
   ```
6. Inspect the active deployment, including the active model version:
   ```bash
   curl http://localhost:8000/status
   ```
7. Inspect MLflow runs and registered model versions at `http://localhost:${MLFLOW_SERVICE_PORT}`. The default external MLflow port is `9010`.
8. Run the smoke test:
   ```bash
   ./scripts/smoke.sh
   ```
   The smoke tests use lighter training overrides and sampled rows so the live path completes faster than the default demo request.
   To run the explicit rollback smoke scenario:
   ```bash
   ./scripts/smoke-rollback.sh
   ```
9. Run fast unit tests from the router project:
   ```bash
   cd router
   uv run pytest tests
   ```

## Demo Walkthrough

For a short walkthrough, the cleanest demo path is:

1. Show the trainer spec in `router/specs/spec.yaml`.
2. Trigger `/admin/train_then_roll/sklearn-model-1` with one hyperparameter override.
   For a faster live walkthrough, include `DATASET_SAMPLE_ROWS` so the full orchestration path stays visible without waiting on the full dataset.
3. In MLflow, show the resulting run, dataset inputs, parameters, evaluation metrics, and registered model version.
4. Show that rollout creates or updates the active serving container only after health checks pass.
5. Hit the stable `/invocations` endpoint through the proxy.
6. Call `/status` to show the active container, model name, model version, and production alias.
7. Mention that `/admin/roll` can promote an existing model version or alias without retraining.

## Creating Your Own Trainer and Serving Images

1. Copy `model-images/sklearn-model-1/` into a new model-specific folder.
2. Update the training code to prepare data, train, evaluate, and log into MLflow for your model.
3. Update the Dockerfiles so `docker compose build` produces trainer and serving images with the tags referenced by the router spec.
4. Add a new trainer entry to `router/specs/spec.yaml`:
   ```yaml
   my-new-model:
     trainer_image: trainer-my-model:latest
     serve_image: server-my-model:latest
     timeout: 3600
     env:
       REGISTERED_MODEL_NAME: MyCoolModel
       MLFLOW_EXPERIMENT: my_experiment
   ```
5. Trigger training through `/admin/train/{spec-name}` or `/admin/train_then_roll/{spec-name}`.

Serving-only promotions can reuse existing registry entries by invoking `/admin/roll` with a model name and version or alias.

## Router API

The control-plane API lives in `router/` and is documented in
[`router/README.txt`](router/README.txt).

The main endpoints are:

- `GET /status` for active deployment state and health
- `POST /admin/train/{trainer}` for synchronous training
- `POST /admin/train_then_roll/{trainer}` for train-and-promote
- `POST /admin/roll` for serving-only promotion of an existing model version or alias
- `POST /admin/rollback` for restoring the previously active deployment

## Design Notes

- Trainer specs map logical workloads to Docker images, environment defaults, and rollout behavior.
- The router creates the MLflow run before training starts so container execution and model registry lineage stay tied together.
- The reference trainer logs dataset-level lineage and standardized evaluation artifacts to MLflow, not just a final model file.
- Rollout uses a blue/green-style candidate container, health check gate, and proxy cutover.
- Active deployment state is persisted to disk so router restarts can validate and recover the last-known active target.
- Successful promotions also persist the previous deployment so rollback can be triggered without manually supplying a model version.
- `/status` exposes both container-level and model-level deployment state for operational checks and demos.

## Example Directory Layout

```text
router/
├── main.py              # FastAPI app wiring status, trainer, and rollout routers
├── specs/spec.yaml      # Trainer specifications
├── trainer/             # Training orchestration and MLflow run management
├── roll/                # Candidate deployment and proxy switching
└── status/              # Active deployment status endpoint
model-images/
└── sklearn-model-1/     # Reference scikit-learn trainer and serving images
```

## Current Limitations

- The reference implementation ships with two bundled model backends: scikit-learn and PyTorch.
- The default environment is local-only: local Docker, local MLflow, SQLite backend store, and file-based artifacts.
- Rollback currently tracks only the immediately previous deployment, not a full deployment history.
- Test coverage is still thin beyond the current happy-path and rollback smoke scenarios.
- The project focuses on control-plane behavior and deployment workflow, not on advanced feature engineering or distributed training.

## Future Improvements

- Expand rollback from a single previous deployment into a richer deployment history.
- Add a more complex second-stage backend, likely transformer-based, to extend the framework-agnostic story beyond tabular regression.
- Add more targeted unit tests around trainer spec resolution and rollout state handling.
- Support remote datasets and artifact stores such as S3-compatible storage.
- Expand promotion policy options beyond direct production alias updates.
