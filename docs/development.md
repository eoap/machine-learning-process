# Developer tooling

Use Python 3.12 and run `task setup` from the repository root. Runtime and
notebook dependencies are hash-locked; local packages are installed separately
without resolving their dependencies. TensorFlow 2.21.0, tf2onnx 1.17.0 and
ONNX Runtime 1.31.0 form the tested export stack. Dependency upgrades require
regenerating locks and passing export-equivalence and actual inference tests.

| Command | Purpose |
| --- | --- |
| `task code:test` | Regression and model handoff tests |
| `task code:check` | Check correctness, imports, modernization and complexity |
| `task code:lint` | Check formatting across application code, helpers and tests |
| `task code:typecheck` | Strict typing for maintained boundary helpers |
| `task code:security` | Check medium/high severity application and helper findings |
| `task cwl:validate` | Validate isolated contracts and generate Markdown |
| `task docs:build` | Strict documentation build |
| `task labs` | Execute all nine notebooks and retain results |

Formatting and lint checks cover all application packages and maintained helpers.
Strict typing covers the maintained boundary helpers listed below; this is not
evidence of full typing coverage for legacy training utilities.

`EOAP_LOCAL_FIXTURE=1` is the default execution fixture. It produces synthetic
EuroSAT-shaped training data and twelve-band inference data, including edge
tiles. These validate software behavior, not classification accuracy. Use
`EOAP_LOCAL_FIXTURE=0`, `EOAP_TRAINING_INPUT`, `EOAP_INFERENCE_INPUT` and a reachable
`MLFLOW_TRACKING_URI` for real-data acceptance. Export and inference share the
selected trained model; they no longer overwrite the bundled model.

Container builds use the repository root, for example:

```bash
podman build -f inference/make-inference/Dockerfile -t localhost/ml-inference:lab .
```

The development-platform module supplies PDE, isolated Calrissian, MLflow and
SeaweedFS. Run the platform tasks with an explicit `KUBE_CONTEXT`. Its initialization
preserves edited checkouts and user files and installs the same locks. AMD64 is
the initial target; ARM64 remains unverified.

The root `pyproject.toml` packages the training, export, inference and MLM Python
modules as `eoap-machine-learning-process`. The adapted `pyproject copy.toml`
contains the same project configuration; use `pyproject.toml` as the canonical
file. Individual module distributions remain available for dedicated images.
The root multi-stage Dockerfile installs a built wheel into a non-root runtime
virtual environment. It requires no DEM auxiliary data or EGMS gateway sources.

Run the required project checks with:

```bash
hatch run dev:check
hatch run dev:typecheck
hatch run dev:security
hatch run test:test
```

Hatch installs the hash-locked development dependencies and local modules.
Ruff checks imports, correctness, simplifications and complexity (maximum 10)
across application code, helpers and tests. Formatting covers those paths too.
Strict typing currently covers the contract, inference-fixture, export, model
metadata and STAC staging helpers; legacy modules are not yet fully typed.
The security gate reports medium/high findings; lower-severity findings remain.

All generated STAC Catalogs, Collections and Items use core STAC **1.1.0**.
Model Items declare the MLM **1.6.0** schema and validate it, including model roles,
artifact type and named EO input bands. The tensor's last dimension is `bands`.
Normalization is recorded as a Python preprocessing expression dividing float32
input by the contract divisor. This uses the required fork's implemented API;
MLM 1.6.0's scaling-array API is not implemented by that fork. The inference
contract continues to provide the numeric divisor consumed by inference.
Inference Items declare Raster metadata and validate their extension schemas.
Fixture creation and STAC staging validate before writing.
Validation needs access to the declared public extension schemas.

PySTAC is the required Terradue `t2_extensions` fork with validation support.
Its exact source revision, wheel hash and license are recorded in
[`vendor/README.md`](https://github.com/eoap/machine-learning-process/blob/main/vendor/README.md).
The wheel enables hash-checked installation without resolving an unhashable Git
reference at runtime; it does not replace the fork with upstream PyPI PySTAC.
