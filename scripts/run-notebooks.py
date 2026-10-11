"""Execute the nine lessons in dependency order and retain notebook evidence."""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def run(output, timeout, scenario=None):
    os.environ["EOAP_LAB_OUTPUT"] = str(output / "outputs")
    report = {
        "candidate": os.environ.get("APPLICATION_PACKAGE_VERSION"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_source": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "runner": os.environ.get("EOAP_CWL_RUNNER", "cwltool"),
        "architecture": {"x86_64": "amd64", "aarch64": "arm64"}.get(
            platform.machine(), platform.machine()
        ),
        "real_data": os.environ.get("EOAP_LOCAL_FIXTURE", "1") == "0",
        "evidence_location": str(output),
        "notebooks": [],
    }

    for path in sorted((ROOT / "practice-labs").rglob("*.ipynb")):
        if scenario and scenario not in path.parts:
            continue
        notebook = nbformat.read(path, as_version=4)
        target = output / path.relative_to(ROOT / "practice-labs")
        target.parent.mkdir(parents=True, exist_ok=True)
        manager = AsyncKernelManager(
            kernel_name="ml_env",
            kernel_spec_manager=KernelSpecManager(
                kernel_dirs=[str(Path(sys.prefix) / "share/jupyter/kernels")]
            ),
        )
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="ml_env",
            km=manager,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        except Exception:
            report["notebooks"].append({"path": str(path.relative_to(ROOT)), "status": "failed"})
            raise
        else:
            report["notebooks"].append({"path": str(path.relative_to(ROOT)), "status": "passed"})
        finally:
            nbformat.write(notebook, target)
            (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Passed {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/notebook-acceptance" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "Alternative1-Application_Steps",
            "Alternative2-Containers",
            "Alternative3-CWL-Workflows",
        ],
    )
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            str(Path(sys.prefix)),
            "--name",
            "ml_env",
        ],
        check=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.output.resolve(), args.timeout, args.scenario)
