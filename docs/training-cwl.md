# Training Module & CWL Runner

This Application Package provides a CWL document containing a top-level workflow with a single `CommandLineTool` step that executes the training pipeline. It also supports **parallel execution**, allowing users to specify multiple sets of hyperparameter or training configurations. This makes it suitable for large-scale experiments and hyperparameter tuning on platforms like a Minikube cluster.

To execute the training workflow, users can choose between [cwltool](https://github.com/common-workflow-language/cwltool) and [Calrissian](https://github.com/Duke-GCB/calrissian) as their CWL runners.

# Inputs:

| Parameter              | Type     | Description |
|------------------------|----------|-------------|
| MLFLOW_TRACKING_URI    | string      | An environment variable for the MLFLOW_TRACKING_URI |
| stac_reference         | string   | URL pointing to a STAC catalog. The model reads GeoParquet annotations from the collection's assets. |
| BATCH_SIZE             | int[]    | Number of batches |
| CLASSES                | int      | Number of land cover classes to classify. |
| DECAY                  | float[]  | Decay value used in training. |
| EPOCHS                 | int[]    | Number of epochs |
| EPSILON                | float[]  | Epsilon value (model hyperparameter) |
| LEARNING_RATE          | float[]  | Initial learning rate for the optimizer. |
| LOSS                   | string[] | Loss function for training. Options: `binary_crossentropy`, `cosine_similarity`, `mean_absolute_error`, `mean_squared_logarithmic_error`, `squared_hinge`. |
| MEMENTUM               | float[]  | Momentum parameter used in optimizers |
| OPTIMIZER              | string[] | Optimization algorithm. Options: `Adam`, `SGD`, `RMSprop`. |
| REGULARIZER            | string[] | Regularization technique to avoid overfitting. Options: `l1l2`, `l1`, `l2`, `None`. |
| SAMPLES_PER_CLASS      | int      | Number of samples to use for training per class. |


## How to execute the Application Ppackage?
Before running the application with a CWL runner, make sure to download and use the latest version of the CWL document:
```
cd training/app-package
VERSION=$(curl -s https://api.github.com/repos/eoap/machine-learning-process/releases/latest | jq -r '.tag_name')
curl -L -o "tile-sat-training.cwl" \
  "https://github.com/eoap/machine-learning-process/releases/download/${VERSION}/tile-sat-training.${VERSION}.cwl"

```


### **Run the Application Package**:
There are two methods to execute the application:


- Executing the tile-based-training using `cwltool` in a terminal:
   ```
    cwltool --podman --debug --parallel tile-sat-training.cwl#tile-sat-training params.yaml
   ```
    


- Executing the tile-based classification using `calrissian` in a terminal:

   ```
    calrissian --debug --stdout /calrissian/out.json --stderr /calrissian/stderr.log --usage-report /calrissian/report.json --parallel --max-ram 10G --max-cores 2 --tmp-outdir-prefix /calrissian/tmp/ --outdir /calrissian/results/ --tool-logs-basepath /calrissian/logs tile-sat-training.cwl#tile-sat-training params.yaml
   ```
   > You can monitor the pod creation using `kubectl` command below:
   >
   >   `kubectl get pods` 


## How the CWL document is designed:
The CWL workflow can be executed using either `cwltool` or `calrissian`. The execution requires a `params.yml` file, which supplies all the necessary inputs defined in the CWL specification. The workflow is structured to run the module according to the diagram outlined below:

![image](imgs/training.png "Training Workflow")

> **`[]`** in the image above indicates that the user may pass a list of parameters to the application package.

## For developers

Users can train multiple tile-based classifiers using the CWL runner, with model weights tracked as artifacts in MLflow. Once training is complete, the next step is to retrieve the best-performing model, based on the chosen evaluation metric, from the MLflow artifact registry and convert it to ONNX format. This process is detailed in the ["Export the Best Model to ONNX Format"](./extract-model.md) guide. The resulting ONNX model can then be integrated into the inference application package.

## Troubleshooting

Users might encounter memory-related issues when executing workflows with CWL Runners (especially with `cwltool`). These issues can often be mitigated by reducing the `ramMax` parameter (e.g. `ramMax: 1000`) specified in the CWL file, which can help prevent excessive memory allocation.