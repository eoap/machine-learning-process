# Describe a trained model

Use STAC 1.1.0 with the MLM 1.6.0 extension to record the exported model's input/output contract,
artifact and source-run provenance. Follow [the model-description lesson](Describe-MLmodel.md)
and MLM/Describe-MLmodel.ipynb using the locked ml_env environment installed by
`task setup`. Export creates the model contract; description validates its checksum and the declared STAC schemas.
