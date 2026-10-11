# Development acceptance — 2026-10-11

The maturity implementation passed local software checks and live AMD64 PDE
acceptance. This is development evidence from an uncommitted source overlay on
`feature/maturity-parity`, based on `f82fe8b6f6924a0e42503e408b3961630c315f16`.
It is not published-candidate or stable-release acceptance.

| Check | Result |
| --- | --- |
| Regression suite | 16 tests passed, including actual ONNX execution and Keras export |
| Developer checks | Correctness lint, maintained formatting, scoped typing and medium/high Bandit passed |
| Contracts and documentation | Both standalone CWL 1.2 `#main` documents validate; strict MkDocs build passes |
| Packaging | Training, inference and MLM wheels build; both command images execute |
| Release tooling | Actionlint passes; local OCI push/pull, annotations and digest-preserving copy checks pass with placeholder image references |
| Processor vulnerability gate | Zero fixable HIGH/CRITICAL findings in both images; no exceptions |
| Synthetic notebook acceptance | All nine lessons passed together in PDE |
| Real-data notebook acceptance | All nine lessons passed together with 100 labelled EuroSAT chips |
| Geospatial/model checks | Selected MLflow artifact retrieval, Keras/ONNX equivalence, source timestamp, raster grid, uint8 classes and nodata checks passed |
| Storage/editor checks | Editor HTTP and S3 put/head/get/delete passed |
| Lifecycle | Workspace and old environment retained; fresh compatible environment and bucket initialized; prior disposable S3 object discarded |

The cluster is `eoap-mastering-app-package-docker`, namespace
`tile-based-training`, architecture AMD64, with Kubernetes/Calrissian execution.
PDE uses its explicit AMD64 pilot image. MLflow 3.17.0 uses the existing proxied
artifact mode; this work does not enable MLflow S3 artifact storage. SeaweedFS
chart 4.48.0 provides dedicated authenticated disposable S3 storage.

The real observations come from [EuroSAT Zenodo record 7711810](https://zenodo.org/records/7711810).
The complete `EuroSAT_MS.zip` matched the publisher's MD5
`091174add3c8e680a49244acf185b9f0`; its SHA256 is
`eb0cf47380b0ff0dba68726bcb9184f8b261efcbb818d3d2b00b557ab0835b33`.
Selection used the first ten sorted IDs for each of the ten classes in the
repository GeoParquet. Inference used twelve bands of one real chip. STAC
timestamps were retained from the supplied metadata. One-epoch training checks
execution and model handoff; it does not establish classification accuracy.

Executed notebooks, models and local raster outputs remain on the editor PVC:

- `/workspace/lab-results/ml-final-nine` — synthetic execution
- `/workspace/lab-results/ml-real-nine` — real-data execution
- `/workspace/lab-results/real-eurosat-inputs` — real input subset and provenance

A host copy of real execution evidence, logs, scans and preparation scripts is
under `runs/acceptance/2026-10-11/`. This directory is intentionally untracked.
S3 publication was verified before lifecycle testing; those disposable objects
were subsequently removed by the tested restart. Local outputs remain retained.

Remaining promotion gates are a clean published source/candidate, registry and
release verification in CI, and acceptance against those exact published
artifacts. Stable publication requires a committed acceptance record and reuses
accepted processor digests. No stable record, release, tag or image was published
by this implementation. ARM64 and the platform-wide image selection strategy
remain unvalidated.


## STAC and packaging follow-up

The example Dockerfile and packaging configuration have been adapted to the
course modules. The follow-up uses Terradue PySTAC with validation, STAC 1.1.0
for generated objects, and MLM 1.6.0 for model metadata. The regression suite
now has 19 passing tests, including schema validation and invalid metadata
rejection. Stronger Ruff rules and complexity checks, formatting, scoped strict
typing, and the configured Bandit severity gate pass through Hatch.

The updated synthetic run passed all nine notebooks on AMD64 using host tools,
Docker and local CWL execution. The bundled root image and dedicated training
and inference images built and executed their commands; all three scans report
zero fixable HIGH/CRITICAL vulnerabilities. The runtime uses UID 1001 and passes
`pip check`. Root wheel, source distribution and all three module wheels build.
Evidence is retained under `runs/acceptance/stac16/`, including the notebook
record, Hatch logs and vulnerability reports.

The real-data Kubernetes evidence above predates this follow-up. It does not
certify this new source overlay or rebuilt images. Published-candidate acceptance
still requires rerunning the documented real-data platform gates.
