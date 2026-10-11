cwlVersion: v1.2
$namespaces:
  s: https://schema.org/
s:softwareVersion: 0.1.0
$graph:
- class: Workflow
  id: main
  label: Tile-SAT Inference on Sentinel-2 data
  doc: Classify Sentinel-2 tiles into the ten EuroSAT land-cover classes; class 10
    is nodata.
  requirements:
    InlineJavascriptRequirement: {}
    ScatterFeatureRequirement: {}
    SubworkflowFeatureRequirement: {}
    SchemaDefRequirement:
      types:
      - $import: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml
  inputs:
    input_reference:
      doc: S2 product
      label: S2 product
      type:
        type: array
        items: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI
    model:
      type:
      - 'null'
      - File
      default: null
      label: Model
      doc: Optional explicit ONNX model; otherwise use the bundled model.
  outputs:
    results:
      outputSource:
      - make_inference/artifacts
      type: Directory[]
      label: results
      doc: results for the inference process.
  steps:
    make_inference:
      in:
        input_reference: input_reference
        model: model
      out:
      - artifacts
      run: '#make_inference'
      scatter:
      - input_reference
      scatterMethod: dotproduct
- class: CommandLineTool
  id: make_inference
  baseCommand:
  - make-inference
  inputs:
    input_reference:
      type: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI
      inputBinding:
        position: 1
        prefix: --input_reference
        valueFrom: $(self.value)
      label: input reference
      doc: input_reference for the inference process.
    model:
      type:
      - 'null'
      - File
      default: null
      label: Model
      doc: Optional ONNX model artifact.
      inputBinding:
        prefix: --model
  outputs:
    artifacts:
      outputBinding:
        glob: .
      type: Directory
      label: artifacts
      doc: artifacts for the inference process.
  requirements:
    InlineJavascriptRequirement: {}
    NetworkAccess:
      networkAccess: true
    ResourceRequirement:
      coresMax: 1
      ramMax: 4096
    SchemaDefRequirement:
      types:
      - $import: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml
    DockerRequirement:
      dockerPull: ghcr.io/eoap/machine-learning-process/inference:0.1.0
    EnvVarRequirement:
      envDef:
        CPL_VSIL_CURL_USE_HEAD: 'NO'
        GDAL_DISABLE_READDIR_ON_OPEN: EMPTY_DIR
        GDAL_HTTP_TIMEOUT: '30'
  label: make inference
  doc: Execute make inference and return its artifacts.
s:codeRepository: https://github.com/eoap/machine-learning-process
s:license: https://spdx.org/licenses/Apache-2.0
s:author:
- class: s:Person
  s:name: Parham Membari
  s:email: parham.membari@terradue.com
