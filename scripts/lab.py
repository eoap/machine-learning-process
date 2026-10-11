"""Shared lab commands; each scenario uses its own retained output directory."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def command(args, cwd, env=None):
    environment = dict(os.environ if env is None else env)
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    subprocess.run([str(a) for a in args], cwd=cwd, env=environment, check=True)


def prepare(mode):
    directory = Path(os.environ.get("EOAP_LAB_OUTPUT", ROOT / "runs/labs")) / mode
    directory.mkdir(parents=True, exist_ok=True)
    local = os.environ.get("EOAP_LOCAL_FIXTURE", "1") == "1"
    if local:
        command(
            [
                sys.executable,
                ROOT / "scripts/create-training-fixture.py",
                "--output",
                directory / "training-input",
            ],
            ROOT,
        )
        command(
            [
                sys.executable,
                ROOT / "scripts/create-fixture.py",
                "--output",
                directory / "inference-input",
            ],
            ROOT,
        )
        training_input = str(directory / "training-input/catalog.json")
        inference_input = str(directory / "inference-input/item.json")
    else:
        training_input = os.environ.get(
            "EOAP_TRAINING_INPUT",
            "https://raw.githubusercontent.com/eoap/machine-learning-process/main/training/app-package/EUROSAT-Training-Dataset/catalog.json",
        )
        inference_input = os.environ["EOAP_INFERENCE_INPUT"]
    if os.environ.get("EOAP_CWL_RUNNER") == "calrissian" and local:
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("stage_fixture", ROOT / "scripts/stage-fixture.py")
        staging = module_from_spec(spec)
        spec.loader.exec_module(staging)
        endpoint = os.environ.get(
            "AWS_ENDPOINT_URL", "http://tile-based-training-s3-seaweedfs-all-in-one:8333"
        )
        prefix = "fixtures/" + uuid.uuid4().hex
        training_input = staging.publish(
            directory / "training-input", endpoint, prefix + "/training"
        )
        inference_input = staging.publish(
            directory / "inference-input", endpoint, prefix + "/inference", entrypoint="item.json"
        )
    settings = {
        "training_input": training_input,
        "inference_input": inference_input,
        "mode": mode,
        "tracking_uri": os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///" + str(directory / "mlflow.db")
        ),
    }
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))
    return directory


def build(mode, kind):
    if mode == "container":
        engine = os.environ.get("EOAP_CONTAINER_ENGINE", "podman")
        command(
            [
                engine,
                "build",
                "-f",
                ROOT
                / kind
                / ("make-ml-model" if kind == "training" else "make-inference")
                / "Dockerfile",
                "-t",
                f"localhost/ml-{kind}:lab",
                ROOT,
            ],
            ROOT,
        )


def run_tool(directory, kind, args):
    settings = json.loads((directory / "settings.json").read_text())
    if settings["mode"] == "container":
        engine = os.environ.get("EOAP_CONTAINER_ENGINE", "podman")
        command(
            [
                engine,
                "run",
                "--rm",
                *(["--userns=keep-id"] if engine == "podman" else []),
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{directory}:{directory}",
                "-w",
                directory,
                "-e",
                f"MLFLOW_TRACKING_URI={settings['tracking_uri']}",
                f"localhost/ml-{kind}:lab",
                *args,
            ],
            directory,
        )
    else:
        environment = dict(os.environ, MLFLOW_TRACKING_URI=settings["tracking_uri"])
        command(args, directory, environment)


def workflow(directory, kind, job):
    selected = os.environ.get("APPLICATION_PACKAGE_VERSION")
    if selected:
        command(
            [
                sys.executable,
                ROOT / "scripts/prepare-release.py",
                "--version",
                selected,
                "--output",
                directory / "release",
            ],
            ROOT,
        )
        cwl = directory / f"release/tile-sat-{kind}.{selected}.cwl"
    else:
        cwl = ROOT / f"{kind}/app-package/tile-sat-{kind}.cwl"
    override = os.environ.get("EOAP_LAB_IMAGE_" + kind.upper())
    if not selected and override:
        document = yaml.safe_load(cwl.read_text())
        for process in document["$graph"]:
            if process["class"] == "CommandLineTool":
                process["requirements"]["DockerRequirement"]["dockerPull"] = override
        cwl = directory / f"{kind}-development.cwl"
        cwl.write_text(yaml.safe_dump(document, sort_keys=False))
    runner = os.environ.get("EOAP_CWL_RUNNER", "cwltool")
    if runner == "cwltool" and not selected:
        document = yaml.safe_load(cwl.read_text())
        for process in document["$graph"]:
            process["requirements"].pop("DockerRequirement", None)
        cwl = directory / f"{kind}-installed-tools.cwl"
        cwl.write_text(yaml.safe_dump(document, sort_keys=False))
    job_path = directory / f"{kind}-job.yaml"
    job_path.write_text(yaml.safe_dump(job, sort_keys=False))
    runner = os.environ.get("EOAP_CWL_RUNNER", "cwltool")
    output = directory / f"{kind}-cwl"
    if runner == "calrissian":
        run_id = directory.name + "-" + kind + "-" + uuid.uuid4().hex
        run_root = Path("/calrissian") / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        args = [
            "calrissian",
            "--max-ram",
            "8G",
            "--max-cores",
            "2",
            "--tmp-outdir-prefix",
            str(run_root / "temporary") + "/",
            "--usage-report",
            str(run_root / "usage.json"),
            "--outdir",
            output,
        ]
    elif runner == "cwltool":
        # Source acceptance executes installed tools; release acceptance uses published images.
        args = ["cwltool", "--podman" if selected else "--no-container", "--outdir", output]
    else:
        raise ValueError("EOAP_CWL_RUNNER must be cwltool or calrissian")
    command([*args, str(cwl) + "#main", job_path], directory)
    return output


def train(directory):
    settings = json.loads((directory / "settings.json").read_text())
    build(settings["mode"], "training")
    if settings["mode"] == "cwl":
        job = yaml.safe_load((ROOT / "training/app-package/params.yaml").read_text())
        for key in [
            "BATCH_SIZE",
            "DECAY",
            "EPOCHS",
            "EPSILON",
            "LEARNING_RATE",
            "LOSS",
            "MEMENTUM",
            "OPTIMIZER",
            "REGULARIZER",
        ]:
            job[key] = job[key][:1]
        job.update(
            MLFLOW_TRACKING_URI={"value": settings["tracking_uri"]},
            stac_reference={"value": settings["training_input"]},
            EPOCHS=[1],
        )
        output = workflow(directory, "training", job)
    else:
        run_tool(
            directory,
            "training",
            [
                "tile-based-training",
                "--stac_reference",
                settings["training_input"],
                "--EPOCHS",
                "1",
                "--BATCH_SIZE",
                "2",
                "--SAMPLES_PER_CLASS",
                "10",
            ],
        )
        output = directory
    manifests = list(output.rglob("model-run.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one model-run manifest, found {len(manifests)}")
    manifest = json.loads(manifests[0].read_text())
    # CWL moves the directory; resolve the model relative to the moved manifest.
    source = manifests[0].parent / "output/training/trained_model.keras"
    assert source.is_file()
    (directory / "selected-model.json").write_text(json.dumps({"source": str(source), **manifest}))
    return source


def export(directory):
    from tile_based_training.export import export_model

    selected = json.loads((directory / "selected-model.json").read_text())
    import mlflow

    mlflow.set_tracking_uri(selected["tracking_uri"])
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{selected['run_id']}/model/trained_model.keras"
    )
    metadata = export_model(selected["source"], directory / "model.onnx", selected["run_id"])
    assert metadata["sha256"] and len(metadata["bands"]) == 12
    from mlm.describe import describe

    describe(directory / "model.onnx", directory / "model-stac.json")
    return metadata


def infer(directory):
    settings = json.loads((directory / "settings.json").read_text())
    build(settings["mode"], "inference")
    if settings["mode"] == "cwl":
        output = workflow(
            directory,
            "inference",
            {
                "input_reference": [{"value": settings["inference_input"]}],
                "model": {"class": "File", "path": str(directory / "model.onnx")},
            },
        )
    else:
        run_tool(
            directory,
            "inference",
            [
                "make-inference",
                "-i",
                settings["inference_input"],
                "--model",
                directory / "model.onnx",
            ],
        )
        output = directory
    catalogs = [
        p
        for p in output.rglob("catalog.json")
        if "training-input" not in p.parts and "mlruns" not in p.parts
    ]
    assert catalogs, "Inference must produce a STAC catalog"
    import numpy as np
    import pystac
    import rasterio
    from make_inference.ml_helper import item_filter_assets

    source_item = pystac.read_file(settings["inference_input"])
    if isinstance(source_item, pystac.Catalog):
        source_item = next(source_item.get_items(recursive=True))
    with rasterio.open(item_filter_assets(source_item)["B02"]) as source:
        source_grid = (source.shape, source.crs, source.transform)

    for path in catalogs:
        for item in pystac.read_file(str(path)).get_items(recursive=True):
            assert item.datetime == source_item.datetime, "Preserve source acquisition metadata"
            for asset in item.assets.values():
                with rasterio.open(asset.get_absolute_href()) as src:
                    assert (src.shape, src.crs, src.transform) == source_grid
                    assert src.nodata == 10 and src.dtypes == ("uint8",)
                    assert set(np.unique(src.read(1))) <= set(range(11))
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("stage_fixture", ROOT / "scripts/stage-fixture.py")
        staging = module_from_spec(spec)
        spec.loader.exec_module(staging)
        for path in catalogs:
            published = staging.publish(path.parent, endpoint, "outputs/" + uuid.uuid4().hex)
            catalog = pystac.read_file(published)
            for item in catalog.get_items(recursive=True):
                for asset in item.assets.values():
                    with rasterio.open(asset.get_absolute_href()) as src:
                        assert src.crs and set(np.unique(src.read(1))) <= set(range(11))
    return catalogs
