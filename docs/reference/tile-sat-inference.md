# tile-sat-inference

Version: 0.1.0

License: https://spdx.org/licenses/Apache-2.0

## main

Classify Sentinel-2 tiles into the ten EuroSAT land-cover classes; class 10 is nodata.

### inputs

| Port | Type | Description |
| --- | --- | --- |
| input_reference | `{'type': 'array', 'items': 'https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI'}` | S2 product |
| model | `['null', 'File']` | Optional explicit ONNX model; otherwise use the bundled model. |

### outputs

| Port | Type | Description |
| --- | --- | --- |
| results | `Directory[]` | results for the inference process. |

## make_inference

Execute make inference and return its artifacts.

### inputs

| Port | Type | Description |
| --- | --- | --- |
| input_reference | `https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI` | input_reference for the inference process. |
| model | `['null', 'File']` | Optional ONNX model artifact. |

### outputs

| Port | Type | Description |
| --- | --- | --- |
| artifacts | `Directory` | artifacts for the inference process. |
