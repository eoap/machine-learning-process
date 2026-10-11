cwlVersion: v1.2
$namespaces:
  s: https://schema.org/
s:softwareVersion: 0.1.0
$graph:
- class: Workflow
  id: main
  label: Tile-based calssifier on EuroSAT data
  doc: Training a CNN on Sentinel-2 data(EuroSAT) to classify small patch of image
    into different landcover classes.
  requirements:
    InlineJavascriptRequirement: {}
    ScatterFeatureRequirement: {}
    SchemaDefRequirement:
      types: &id001
      - $import: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml
      - name: '#Optimizer'
        type: enum
        symbols:
        - Adam
        - SGD
        - RMSprop
      - name: '#Regularizer'
        type: enum
        symbols:
        - None
        - l1
        - l2
        - l1l2
  inputs:
    MLFLOW_TRACKING_URI:
      label: MLFLOW_TRACKING_URI
      type: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI
      doc: MLFLOW_TRACKING_URI for the training process.
    stac_reference:
      label: stac_reference
      doc: STAC Item label url
      type: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI
    BATCH_SIZE:
      label: BATCH_SIZE
      default:
      - 4
      doc: BATCH_SIZE- model metadata
      type: int[]
    CLASSES:
      label: CLASSES
      default: 10
      doc: CLASSES- model metadata
      type: int
    DECAY:
      label: DECAY
      default:
      - 0.1
      doc: DECAY- model metadata
      type: float[]
    EPOCHS:
      label: EPOCHS
      default:
      - 5
      doc: EPOCHS- model metadata
      type: int[]
    EPSILON:
      label: EPSILON
      default:
      - 2.0e-06
      doc: EPSILON- model metadata
      type: float[]
    LEARNING_RATE:
      label: LEARNING_RATE
      default:
      - 0.0001
      doc: LEARNING_RATE- model metadata
      type: float[]
    LOSS:
      label: LOSS
      default:
      - categorical_crossentropy
      doc: LOSS- model metadata
      type: string[]
    MEMENTUM:
      label: MEMENTUM
      default:
      - 0.95
      doc: MEMENTUM- model metadata
      type: float[]
    OPTIMIZER:
      label: OPTIMIZER
      default:
      - Adam
      doc: OPTIMIZER- model metadata
      type:
        type: array
        items: '#Optimizer'
    REGULARIZER:
      label: REGULARIZER
      default:
      - None
      doc: REGULARIZER- model metadata
      type:
        type: array
        items: '#Regularizer'
    SAMPLES_PER_CLASS:
      label: SAMPLES_PER_CLASS
      default: 500
      doc: SAMPLES_PER_CLASS- model metadata
      type: int
  outputs:
    artifacts:
      outputSource:
      - training/artifacts
      type: Directory[]
      label: artifacts
      doc: artifacts for the training process.
  steps:
    training:
      run: '#training'
      in:
        MLFLOW_TRACKING_URI: MLFLOW_TRACKING_URI
        stac_reference: stac_reference
        BATCH_SIZE: BATCH_SIZE
        CLASSES: CLASSES
        DECAY: DECAY
        EPOCHS: EPOCHS
        EPSILON: EPSILON
        LEARNING_RATE: LEARNING_RATE
        LOSS: LOSS
        MEMENTUM: MEMENTUM
        OPTIMIZER: OPTIMIZER
        REGULARIZER: REGULARIZER
        SAMPLES_PER_CLASS: SAMPLES_PER_CLASS
      scatter:
      - BATCH_SIZE
      - DECAY
      - EPOCHS
      - EPSILON
      - LEARNING_RATE
      - LOSS
      - MEMENTUM
      - OPTIMIZER
      - REGULARIZER
      scatterMethod: dotproduct
      out:
      - artifacts
- class: CommandLineTool
  id: training
  requirements:
    InlineJavascriptRequirement: {}
    NetworkAccess:
      networkAccess: true
    EnvVarRequirement:
      envDef:
        MLFLOW_TRACKING_URI: $(inputs.MLFLOW_TRACKING_URI.value)
        CPL_VSIL_CURL_USE_HEAD: 'NO'
        GDAL_DISABLE_READDIR_ON_OPEN: EMPTY_DIR
        GDAL_HTTP_TIMEOUT: '30'
    ResourceRequirement:
      coresMax: 1
      ramMax: 4096
    SchemaDefRequirement:
      types: *id001
    DockerRequirement:
      dockerPull: ghcr.io/eoap/machine-learning-process/training:0.1.0
  baseCommand:
  - tile-based-training
  arguments: []
  inputs:
    MLFLOW_TRACKING_URI:
      type: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI
      label: MLFLOW TRACKING URI
      doc: MLFLOW_TRACKING_URI for the training process.
    stac_reference:
      type: https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml#URI
      inputBinding:
        prefix: --stac_reference
        valueFrom: $(self.value)
      label: stac reference
      doc: stac_reference for the training process.
    BATCH_SIZE:
      type: int
      inputBinding:
        prefix: --BATCH_SIZE
      label: BATCH SIZE
      doc: BATCH_SIZE for the training process.
    CLASSES:
      type: int
      inputBinding:
        prefix: --CLASSES
      label: CLASSES
      doc: CLASSES for the training process.
    DECAY:
      type: float
      inputBinding:
        prefix: --DECAY
      label: DECAY
      doc: DECAY for the training process.
    EPOCHS:
      type: int
      inputBinding:
        prefix: --EPOCHS
      label: EPOCHS
      doc: EPOCHS for the training process.
    EPSILON:
      type: float
      inputBinding:
        prefix: --EPSILON
      label: EPSILON
      doc: EPSILON for the training process.
    LEARNING_RATE:
      type: float
      inputBinding:
        prefix: --LEARNING_RATE
      label: LEARNING RATE
      doc: LEARNING_RATE for the training process.
    LOSS:
      type: string
      inputBinding:
        prefix: --LOSS
      label: LOSS
      doc: LOSS for the training process.
    MEMENTUM:
      type: float
      inputBinding:
        prefix: --MEMENTUM
      label: MEMENTUM
      doc: MEMENTUM for the training process.
    OPTIMIZER:
      type: '#Optimizer'
      inputBinding:
        prefix: --OPTIMIZER
      label: OPTIMIZER
      doc: OPTIMIZER for the training process.
    REGULARIZER:
      type: '#Regularizer'
      inputBinding:
        prefix: --REGULARIZER
      label: REGULARIZER
      doc: REGULARIZER for the training process.
    SAMPLES_PER_CLASS:
      type: int
      inputBinding:
        prefix: --SAMPLES_PER_CLASS
      label: SAMPLES PER CLASS
      doc: SAMPLES_PER_CLASS for the training process.
  outputs:
    artifacts:
      outputBinding:
        glob: .
      type: Directory
      label: artifacts
      doc: artifacts for the training process.
  label: training
  doc: Execute training and return its artifacts.
s:codeRepository: https://github.com/eoap/machine-learning-process
s:license: https://spdx.org/licenses/Apache-2.0
s:author:
- class: s:Person
  s:name: Parham Membari
  s:email: parham.membari@terradue.com
