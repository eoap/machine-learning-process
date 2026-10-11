"""Publish reachable STAC objects with signed HTTP links for processing pods."""

import json
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pystac
from botocore.config import Config


def _local_href(root: Path, document: Path, href: str | None) -> Path | None:
    """Resolve an existing local reference contained in the fixture directory."""
    if href and not urlparse(href).scheme:
        target = (document.parent / href).resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    return None


def _references(document: pystac.STACObject) -> list[pystac.Link | pystac.Asset]:
    """Return links and assets without resolving remote STAC objects."""
    references: list[pystac.Link | pystac.Asset] = list(document.links)
    if isinstance(document, (pystac.Item, pystac.Collection)):
        references.extend(document.assets.values())
    return references


def _reachable_files(
    root: Path, entrypoint: str
) -> tuple[set[Path], dict[Path, pystac.STACObject]]:
    """Find the referenced tree, preserving non-STAC JSON assets as opaque files."""
    pending = [root / entrypoint]
    files: set[Path] = set()
    documents: dict[Path, pystac.STACObject] = {}
    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.add(path)
        if path.suffix != ".json":
            continue
        try:
            document = pystac.read_file(str(path))
        except pystac.STACTypeError:
            continue
        documents[path] = document
        for reference in _references(document):
            target = _local_href(root, path, reference.href)
            if target:
                pending.append(target)
    return files, documents


def publish(
    root: str | Path,
    endpoint: str,
    prefix: str,
    bucket: str = "results",
    entrypoint: str = "catalog.json",
) -> str:
    """Upload a validated STAC 1.1.0 tree with expiring signed references.

    Args:
        root: Local fixture directory; unrelated files are excluded.
        endpoint: Local lab S3 endpoint using the lab's test credentials.
        prefix: Destination object prefix.
        bucket: Existing destination bucket.
        entrypoint: Relative STAC entrypoint within the fixture directory.

    Returns:
        Signed entrypoint URL, valid for two hours.

    Raises:
        ValueError: The entrypoint lies outside the fixture directory.
        pystac.STACValidationError: A staged STAC document fails validation.
    """
    root = Path(root).resolve()
    if not (root / entrypoint).resolve().is_relative_to(root):
        raise ValueError("Entrypoint must be inside the fixture directory")
    pystac.set_stac_version("1.1.0")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    files, documents = _reachable_files(root, entrypoint)

    def signed(path: Path) -> str:
        key = prefix + "/" + path.relative_to(root).as_posix()
        url = client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=7200
        )
        if not isinstance(url, str):
            raise ValueError("S3 did not return a signed URL")
        return url

    # Validate the entire tree before uploading any objects.
    contents: dict[Path, bytes] = {}
    for path, document in documents.items():
        for reference in _references(document):
            target = _local_href(root, path, reference.href)
            if target:
                if isinstance(reference, pystac.Link):
                    reference.target = signed(target)
                else:
                    reference.href = signed(target)
        # Signed roots are not uploaded yet; preserve the already absolute links
        # without resolving them through a remote catalog during serialization.
        payload = document.to_dict(transform_hrefs=False)
        pystac.validation.validate_dict(payload)
        contents[path] = json.dumps(payload).encode()
    for path in sorted(files):
        client.put_object(
            Bucket=bucket,
            Key=prefix + "/" + path.relative_to(root).as_posix(),
            Body=contents[path] if path in contents else path.read_bytes(),
            ContentType="application/json"
            if path.suffix == ".json"
            else "application/octet-stream",
        )
    return signed(root / entrypoint)
