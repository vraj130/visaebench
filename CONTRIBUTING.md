# Contributing

Thanks for your interest in `visaebench`. This library is evaluation-only: it
scores Sparse Autoencoders trained on Vision Transformer patch activations. It
does not include SAE training code, and pull requests that add training or new
metrics are out of scope for now.

## Development setup

Install the package in editable mode with the development extras (pytest,
pytest-cov, build, twine):

```bash
git clone https://github.com/vraj130/visaebench && cd visaebench
python -m pip install -e ".[dev]"
```

## Running the tests

```bash
pytest                                  # full suite
pytest tests/test_metrics/test_fvu.py   # one file
pytest -k fvu                           # match by name
```

The default `pytest` invocation runs only CPU tests that need no network and no
GPU. Two markers gate the rest:

- `gpu`: tests that require a CUDA device.
- `network`: tests that download weights or datasets (HuggingFace, ImageNet).

Both are excluded by default via `addopts` in `pyproject.toml`. To run them
explicitly:

```bash
pytest -m "gpu or network"
```

When adding tests, keep the default suite CPU-only and offline. Use small random
tensors, pass `device="cpu"` explicitly (several signatures default to
`"cuda"`), and mark anything that needs a GPU or downloads with `@pytest.mark.gpu`
or `@pytest.mark.network` so continuous integration keeps skipping it.

## Releasing

Maintainer steps to publish a new `visaebench` release to PyPI. The project uses
PyPI Trusted Publishing (OIDC), so no API tokens are stored in the repo. The
`publish.yml` GitHub Actions workflow builds and uploads the distributions when a
GitHub release is created.

### One-time setup (first release only)

1. Create the PyPI project by registering it as a Trusted Publisher on PyPI
   (this reserves the name without a manual first upload):
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new pending publisher with:
     - PyPI project name: `visaebench`
     - Owner: `vraj130`
     - Repository name: `visaebench`
     - Workflow name: `publish.yml`
     - Environment name: `pypi`
2. In the GitHub repo settings, create an environment named `pypi` (Settings
   → Environments → New environment) so the workflow's `environment: pypi`
   matches the Trusted Publisher configuration.

### Per-release steps

1. Confirm the version. Edit `visaebench/version.py` so `__version__` is the
   new release version (e.g. `"0.1.0"`). The build reads this dynamically via
   `[tool.setuptools.dynamic]` in `pyproject.toml`.
2. Update `CHANGELOG.md`: move the pending changes under a new dated version
   heading and update the compare/tag link at the bottom.
3. Commit these changes and merge them to `main`.
4. Push the release tag matching the version:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
5. Create the GitHub release for that tag (GitHub UI → Releases → Draft a new
   release → choose tag `v0.1.0` → publish). Publishing the release triggers
   `publish.yml`.
6. Verify the workflow run under the Actions tab completed successfully and
   uploaded the sdist and wheel to PyPI.
7. Final clean-environment check:
   ```bash
   python -m venv /tmp/visaebench-release-check
   /tmp/visaebench-release-check/bin/pip install visaebench
   /tmp/visaebench-release-check/bin/python -c "import visaebench; print(visaebench.__version__)"
   ```

### Local build sanity check (optional, before tagging)

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```
