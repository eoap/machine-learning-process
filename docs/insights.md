# Lessons Learned from Building a Machine Learning process for Geospatial Data 

## Introduction

This page highlights the technical challenges, design decisions, and key insights gained while developing the machine learning process for geospatial data pipeline. 

It also includes recommendations for future improvements and practical advice for replicating or extending the setup.

## Design Decisions

### Modular Workflow Templates

* Decision: Separate the CWL execution, training pipeline, and inference pipeline  into distinct workflow templates.

* Outcome: Enhanced reusability for other geospatial pipelines requiring similar preprocessing steps.

### STAC Integration

* Decision: Leverage the STAC API, Geoparquet, and DuckDB for querying and storing geospatial data.
* Outcome: Improved interoperability with other geospatial tools and standards.

### Tracking the process

* Decision: Use MLFLOW exclusively for tracking the process of training workflow and selecting the best model candidate.

### Test inference with Sentinel-2 product
* Decision: Use Stars tool to stage-in a sentinel-2 product ready to pass to inference module.

## Challenges and Solutions

### Build Docker Images

* Challenge: Initially, we used an [advanced tooling technique](https://github.com/eoap/advanced-tooling) that leveraged **Taskfile** to build a Kaniko-based image and reference the CWL files. The image was then pushed to [ttl.sh](https://ttl.sh/), a temporary image registry. This helps to execute the application packages using `calrissian`. However, this process was slow and hard to debug, often failing due to the large size of the Kaniko images.
* Solution: We now push the Docker images to a dedicated GitHub Container Registry.



