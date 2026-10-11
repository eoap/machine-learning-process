# Export a selected model to ONNX

Use the locked environment and select an explicit source or MLflow run. The
exporter checks the ONNX structure and compares two deterministic samples with
the Keras model. It writes model.json beside model.onnx with SHA256 checksums,
band order, normalization, class mapping and source-run provenance.

```bash
export-ml-model --source output/training/trained_model.keras --output runs/model.onnx
# Alternatively use an explicitly selected MLflow run:
export-ml-model --run-id "$MLFLOW_RUN_ID" --tracking-uri "$MLFLOW_TRACKING_URI" --output runs/model.onnx
make-inference --input_reference "$STAC_ITEM" --model runs/model.onnx
```

The training command records the selected run and model path in model-run.json.
The extraction notebooks use that manifest rather than ranking mutable runs or
requiring an arbitrary accuracy threshold. Synthetic fixture accuracy is not
representative of real EO classification quality.
