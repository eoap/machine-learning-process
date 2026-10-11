# Training with CWL

The standalone CWL v1.2 workflow has a `#main` entrypoint and scatters matched
hyperparameter arrays using dotproduct. Every scatter array must have the same
length. Optimizer and regularizer values are enums; references are URI records.
See [the generated contract](reference/tile-sat-training.md) for all ports.

```yaml
MLFLOW_TRACKING_URI:
  value: http://my-mlflow:5000
stac_reference:
  value: https://raw.githubusercontent.com/eoap/machine-learning-process/main/training/app-package/EUROSAT-Training-Dataset/catalog.json
BATCH_SIZE: [2]
CLASSES: 10
DECAY: [0.1]
EPOCHS: [1]
EPSILON: [0.000001]
LEARNING_RATE: [0.0001]
LOSS: [categorical_crossentropy]
MEMENTUM: [0.95]
OPTIMIZER: [Adam]
REGULARIZER: [None]
SAMPLES_PER_CLASS: 10
```

Training writes `output/training/trained_model.keras` and `model-run.json` in each
result directory and logs the raw Keras artifact, parameters and evaluation
metrics to MLflow. Export an explicitly selected run using [the export guide](extract-model.md).

## Execute a selected release

Set `APPLICATION_PACKAGE_VERSION` to a published candidate or stable tag. The
preparation script checks its CWL metadata, checksum, entrypoint and immutable
container requirements and writes to a runtime directory without modifying source.

```bash
python scripts/prepare-release.py --version "$APPLICATION_PACKAGE_VERSION" --output runs/release
cwltool --podman --outdir runs/training-result "runs/release/tile-sat-training.${APPLICATION_PACKAGE_VERSION}.cwl#main" training/app-package/params.yaml
```

The platform's notebook tasks can instead use Calrissian with the same selected
release. Configure the target cluster, resource limits and accessible inputs
before execution. Measure memory usage with representative inputs rather than
reducing memory requests without checking the workload.
