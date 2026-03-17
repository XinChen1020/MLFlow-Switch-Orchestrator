# Demo Data

`demo_data/diabetes.csv` is the small checked-in dataset used by the bundled
`sklearn-model-1` and `pytorch-model-1` trainer images.

Why it exists:

- the demo training flow is deterministic out of the box
- the training images do not need to generate data at runtime
- reviewers can inspect the exact CSV used by the default walkthrough

To regenerate the file from `sklearn.datasets.load_diabetes`:

```bash
cd model-images/sklearn-model-1
uv run python ../../demo_data/prepare_demo_data.py
```

The trainers still accept arbitrary datasets by overriding `DATASET_PATH` and
`TARGET_COLUMN`.
