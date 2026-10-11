# Release acceptance

The manual release workflow publishes candidates from a feature/development
branch and stable releases only from main/master. PR checks install locked
packages, run regressions, validate CWL/documentation and build wheels/images.
The release workflow repeats these gates and blocks fixable HIGH/CRITICAL image
vulnerabilities. No vulnerability exception is included.

Publication refuses existing tags, serializes releases, pins images by digest,
and serializes standalone source graphs directly. It does not use a CWL packing
command. Module-defined types remain embedded. OCI packages are pulled by digest,
compared byte-for-byte, checked against their annotations and validated again.
A draft GitHub release receives checksummed CWL and OCI inventories before
publication. Registry operations are not transactional; failures may leave images
or artifacts, so inspect the failed run and use a new version.

Before stable promotion, execute all nine notebooks against the published
candidate on the target PDE cluster. Use `APPLICATION_PACKAGE_VERSION` and
`EOAP_CWL_RUNNER=calrissian`; provide training/inference inputs readable by the
processing pods. Editor-local fixture paths require shared accessible mounts or
STAC publication before Kubernetes processing. Local fixture acceptance does
not replace the real-data acceptance path.

Record exact source tree, candidate version, image/package digests, cluster,
architecture, notebook results, model checksum, numerical equivalence, STAC/raster
checks and MLflow artifact retrieval. Run editor/S3 smoke tests and workspace,
environment and disposable-S3 lifecycle checks after workloads finish. The
platform's shared lifecycle helper preserves user workspaces and backs up the old
environment. Keep executed notebooks and logs under the versioned evidence path.

Stable releases require a committed acceptance record checked by the release
workflow. The record must identify a clean source checkout and each of the nine
distinct lesson paths. Development overlays do not qualify. Environment
protection can additionally require reviewers; configuring
GitHub environment protection is an account-level action outside these files.
Stable publication copies the accepted processor manifests and verifies that
their digests are unchanged. CWL packages are regenerated with the stable
version and the same accepted images, then verified after publication.
