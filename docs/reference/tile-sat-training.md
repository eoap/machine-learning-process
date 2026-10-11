# tile-sat-training

Version: 0.1.0

License: https://spdx.org/licenses/Apache-2.0

## main

Training a CNN on Sentinel-2 data(EuroSAT) to classify small patch of image into different landcover classes.

### inputs

| Port | Type | Description |
| --- | --- | --- |
| MLFLOW_TRACKING_URI | `https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI` | MLFLOW_TRACKING_URI for the training process. |
| stac_reference | `https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI` | STAC Item label url |
| BATCH_SIZE | `int[]` | BATCH_SIZE- model metadata |
| CLASSES | `int` | CLASSES- model metadata |
| DECAY | `float[]` | DECAY- model metadata |
| EPOCHS | `int[]` | EPOCHS- model metadata |
| EPSILON | `float[]` | EPSILON- model metadata |
| LEARNING_RATE | `float[]` | LEARNING_RATE- model metadata |
| LOSS | `string[]` | LOSS- model metadata |
| MEMENTUM | `float[]` | MEMENTUM- model metadata |
| OPTIMIZER | `{'type': 'array', 'items': '#Optimizer'}` | OPTIMIZER- model metadata |
| REGULARIZER | `{'type': 'array', 'items': '#Regularizer'}` | REGULARIZER- model metadata |
| SAMPLES_PER_CLASS | `int` | SAMPLES_PER_CLASS- model metadata |

### outputs

| Port | Type | Description |
| --- | --- | --- |
| artifacts | `Directory[]` | artifacts for the training process. |

## training

Execute training and return its artifacts.

### inputs

| Port | Type | Description |
| --- | --- | --- |
| MLFLOW_TRACKING_URI | `https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI` | MLFLOW_TRACKING_URI for the training process. |
| stac_reference | `https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI` | stac_reference for the training process. |
| BATCH_SIZE | `int` | BATCH_SIZE for the training process. |
| CLASSES | `int` | CLASSES for the training process. |
| DECAY | `float` | DECAY for the training process. |
| EPOCHS | `int` | EPOCHS for the training process. |
| EPSILON | `float` | EPSILON for the training process. |
| LEARNING_RATE | `float` | LEARNING_RATE for the training process. |
| LOSS | `string` | LOSS for the training process. |
| MEMENTUM | `float` | MEMENTUM for the training process. |
| OPTIMIZER | `#Optimizer` | OPTIMIZER for the training process. |
| REGULARIZER | `#Regularizer` | REGULARIZER for the training process. |
| SAMPLES_PER_CLASS | `int` | SAMPLES_PER_CLASS for the training process. |

### outputs

| Port | Type | Description |
| --- | --- | --- |
| artifacts | `Directory` | artifacts for the training process. |
