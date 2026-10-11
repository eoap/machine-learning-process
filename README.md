# Machine-learning-process
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![stability-wip](https://img.shields.io/badge/stability-wip-lightgrey.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#work-in-progress)


Users should follow the documentation available at this [link](https://eoap.github.io/machine-learning-process/).

The documentation is organized into sections, each providing specific information about the project:

- **[Introduction](https://eoap.github.io/machine-learning-process/):** Offers a high-level overview of the project.
- **[How-to Guides](https://eoap.github.io/machine-learning-process/hands-on/):** Outlines the order in which modules should be executed.
- **Tutorials:** Walks through the steps required to run the application packages:
  - **Application Packages:**
    - [Run the Training Module using CWL Runner](https://eoap.github.io/machine-learning-process/training-cwl/)
    - [Run the Inference Module using CWL Runner](https://eoap.github.io/machine-learning-process/inference-cwl/)
    - [Describe a Trained Machine Learning Model](https://eoap.github.io/machine-learning-process/mlm/)
- **Reference Guides:** Provides in-depth documentation for the two application packages:
  - [Training a Machine Learning Model](https://eoap.github.io/machine-learning-process/training-container/)
  - [Running Inference on Real-World Data](https://eoap.github.io/machine-learning-process/inference-container/)
- **[Technical Insights and Learnings](https://eoap.github.io/machine-learning-process/insights/):** Shares the decisions and insights gained during development.

## Development and acceptance

Run `task setup`, `task code:test`, `task cwl:validate`, and `task docs:build`.
See [developer tooling](docs/development.md) and [release acceptance](docs/release.md).
Verified development results are recorded in [acceptance evidence](docs/acceptance.md).
The nine lessons retain their outputs and execute training, model export and
inference in application, container and CWL scenarios. Synthetic fixtures are
execution checks; real-data/cluster acceptance and published artifacts are
separate release gates. Version 0.1.0 is unreleased preparation.
