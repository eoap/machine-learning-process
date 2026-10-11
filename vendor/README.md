The PySTAC wheel is built without modifications from the Terradue
`t2_extensions` branch, commit `5941d0c361d409d35e0aeb36f6ae05b03008de71`.
It is included so runtime installations can use `--require-hashes`; pip cannot
hash-lock a Git dependency. Project metadata retains the required Git dependency.

Wheel SHA256: `94d239298dd5d5e518cb024ef56991c2c0027b2c0262a6f5c5e938300ef5689c`.
Upstream license: [PYSTAC-LICENSE](PYSTAC-LICENSE).

To rebuild, check out that commit and run `python -m build --wheel` in the fork.
Changing the fork revision requires rebuilding the wheel and all dependency locks.
The fork's MLM implementation defaults to 1.4.0; `mlm.describe` uses its existing
field APIs with a schema-version adapter and validates against MLM 1.6.0.
