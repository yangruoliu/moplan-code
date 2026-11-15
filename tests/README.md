# Tests

CPU tests for the release code. They need the Python environment from [docs/INSTALL.md](../docs/INSTALL.md)
(pytest is pinned in `requirements.txt`), but no GPU, model weights, datasets or network access.

Run them from the repository root:

```bash
python -m pytest tests -q
python -m pytest tests/test_setup.py -q      # one module
```

`tests/conftest.py` puts the repository root on `sys.path`, so the tests run from a plain checkout without
`PYTHONPATH`. The docstring at the top of each test module says what it covers. For example, `test_setup.py` checks
the installation files: the package pins in `requirements.txt` and `environment.yml`, `setup/checksums.json`, the
offline code paths of `setup/download_weights.py` and `setup/check_env.py`, the shell scripts under `setup/` and the
`.gitignore` rules. `test_integration.py` checks the components against each other: the weight checksums against
`tools/weights_manifest.json`, the config paths the code names, the flags the launchers pass to each script, the
showcase commands against `generate.py`, and the showcase helper (case numbers, the README case table, and verify on
references that are evaluation exports).

A test that needs an optional package skips itself when the package is missing (for example `h5py`, which only the
Stage-1 data-building code uses).

These tests do not run the models. The GPU smoke tests are separate scripts that need the weights; they are
described in the training and inference documentation (`launch/smoke_*.sh`).
