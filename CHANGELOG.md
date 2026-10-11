# Changelog

## [0.1.0] — unreleased

- Build Rocky Linux 10.1 wheel-based runtime images as UID/GID 2000, validate dependencies and remove runtime pip.
- Align nested packages with the shared hash-locked Hatch quality environment.
- Declare and hash-lock Python 3.12 training, inference and notebook environments.
- Add explicit model export with provenance and Keras/ONNX equivalence checks.
- Accept an ONNX model input and classify partial edge tiles while preserving nodata.
- Preserve unrelated user rasters and move training outputs outside package sources.
- Preserve source timestamps and validate output grids, class values and nodata.
- Introduce typed standalone CWL contracts with a default main entrypoint.
- Add synthetic fixtures, regression tests, executable lessons and verified release tooling.

Breaking changes: CWL references are published URI records with a value field, entrypoints
are #main, and container builds use the repository root as context. Model class
10 represents nodata; EuroSAT classes remain 0–9. This is release preparation,
not a claim of published artifacts or accepted candidate releases.
