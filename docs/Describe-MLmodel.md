# Describe an exported model

Run MLM/Describe-MLmodel.ipynb in the locked ml_env kernel after training/export.
The helper verifies the artifact checksum before creating and schema-validating a STAC 1.1.0 Item with MLM 1.6.0.

```python
from mlm.describe import describe
item = describe("runs/labs/app/model.onnx", "runs/labs/app/model-stac.json")
```

The description includes the twelve Sentinel-2 bands in model order, NHWC input
shape, float32 datatype, a preprocessing expression scaling reflectance by 1/10000, ten EuroSAT output
classes, source MLflow run, model checksum, and ONNX/model-contract assets.
Setting EOAP_MODEL selects another export with an adjacent JSON contract.
The core regression suite exercises this description and rejects modified model
bytes. The helper does not imply an independent scientific accuracy assessment.
