# Demo Data

This directory holds the checked-in demo dataset used by the bundled trainers.

`demo_data/diabetes.csv` is the small checked-in dataset used by the bundled
`sklearn-model-1` and `pytorch-model-1` trainer images.

To regenerate the file from `sklearn.datasets.load_diabetes`, run this from the
repo root:

```bash
cd model-images/sklearn-model-1 && uv run python ../../demo_data/prepare_demo_data.py
```

The trainers still accept arbitrary datasets by overriding `DATASET_PATH` and
`TARGET_COLUMN`. For faster demos or smoke tests, they also accept
`DATASET_SAMPLE_ROWS` to train on a deterministic subset of the same CSV.
