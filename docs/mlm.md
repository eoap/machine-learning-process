# Describes a trained machine learning model 
This Item describe a trained machine learning model using [MLM](https://github.com/stac-extensions/mlm) STAC extension. The STAC Machine Learning Model (MLM) Extension provides a standard set of fields to describe machine learning models trained on overhead imagery and enable running model inference.

The main objectives of the extension are:

- to enable building model collections that can be searched alongside associated STAC datasets
- record all necessary bands, parameters, modeling artifact locations, and high-level processing steps to deploy an inference service.

For additional information please follow this [Describe-MLmodel](./Describe-MLmodel.md) notebook.

## For developers:
To run the notebook successfully, you must install the dependencies with hatch:

```
hatch shell prod
hatch -e prod run python -m ipykernel install --user --name=mlm --display-name "mlm"
```


