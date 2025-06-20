# Machine Learning Process


## Introduction

This learning resource demonstrates a Machine Learning (ML) system for classification of Sentinel-2 images into 10 different classes using cloud-native technologies. The system leverages MLFLOW to track the training process and selects the best candidate trained model from MLFLOW server. 

There are two workflows developed: one for training a deep learning model classifier on EuroSAT dataset, and one for running a prediction on a real world Sentinel-2 data.  The automation is achieved using Kubernetes-native tools, making the setup scalable, modular, and suitable for Earth Observation and geospatial applications.



## Key Components

This setup integrates the following technologies and concepts:

### MLFLOW

* Manages end-to-end ML workflows, from development to production
* End-to-end MLOps solution for traditional ML, including integrations with traditional ML models, and Deep learning one
* Simple, low-code performance tracking with autologging
* State-of-the-art UI for model analysis and comparison

### STAC & DuckDB

* Acts as the primary data source by providing geospatial data in a standardized format. The STAC collection includes references to the EUROSAT benchmark dataset, which can be queried using DuckDB.

## High-Level Architecture

The system is designed to handle the following flow:

1. Training pipeline: A CNN model trained on [EuroSAT](https://github.com/phelber/EuroSAT) dataset which already exist on a dedicated STAC endpoint. The MLFLOW tracks the whole process to monitor the life cycle of training.

2. Inference: Run the inference pipeline to perform tile-based classification on Sentinel-2 L1C products.

4. Workflow Execution: Both training and inference pipeline will be executed using the CWL-based algorithm.


## Why Use This Setup?

This setup demonstrates the power of combining machine learning paradigms with container-native workflows to enable scalable geospatial analysis.

It is particularly suited for Earth observation and scientific workflows because:

* Scalability: Kubernetes ensures workflows can handle varying loads effectively.
* Modularity: Components can be easily reused or replaced for other applications.
* Automation: Events trigger workflows without manual intervention, enabling real-time processing.

Through this resource, you'll learn to implement a cloud-native pipeline for tile-based classification, which can be extended to other geospatial or scientific applications.




