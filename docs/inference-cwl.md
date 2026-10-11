# Inference with CWL

The standalone CWL v1.2 workflow has a `#main` entrypoint. It scatters STAC item
URI records and accepts an optional ONNX model File. Omitting the model selects
the bundled EuroSAT model. See [the generated contract](reference/tile-sat-inference.md).

```yaml
input_reference:
  - value: /absolute/path/to/item.json
model:
  class: File
  path: /absolute/path/to/model.onnx
```

For Kubernetes, URI references must be readable by processing pods. The platform
lab helper publishes fixtures through authenticated SeaweedFS with signed links.
Outputs are self-contained STAC catalogs and georeferenced classification rasters,
with classes 0–9 and nodata 10. Partial edge tiles are padded for inference and
cropped back to the original extent.

## Execute a selected release

Set `APPLICATION_PACKAGE_VERSION` to a published candidate or stable tag. The
preparation script checks its CWL metadata, checksum, entrypoint and immutable
container requirements and writes to a runtime directory without modifying source.

```bash
python scripts/prepare-release.py --version "$APPLICATION_PACKAGE_VERSION" --output runs/release
cwltool --podman --outdir runs/inference-result "runs/release/tile-sat-inference.${APPLICATION_PACKAGE_VERSION}.cwl#main" inference/app-package/params.yml
```

The platform's notebook tasks can instead use Calrissian with the same selected
release. Configure the target cluster, resource limits and accessible inputs
before execution. Measure memory usage with representative inputs rather than
reducing memory requests without checking the workload.
