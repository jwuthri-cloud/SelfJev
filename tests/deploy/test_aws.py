"""The AWS deployment's first-boot script and presets (no AWS call is made)."""

from selfjev.deploy.aws import BASE_MODEL, MACHINES, PORT, user_data


def test_user_data_installs_the_pinned_ref_and_serves_behind_the_key():
    script = user_data("sj-key", "v0.2.0", max_hours=2)
    assert script.startswith("#!/bin/bash") and "shutdown -h +120" in script
    assert "git checkout v0.2.0" in script and "GIT_LFS_SKIP_SMUDGE=1 git clone" in script
    assert "git lfs pull --include 'weights/selfjev_4b/*'" in script  # only the served adapter, not the whole LFS store
    assert "uv sync --frozen --no-dev --extra serve --extra gpu" in script and BASE_MODEL[1] in script
    assert "Environment=SELFJEV_API_KEYS=sj-key" in script and f"selfjev serve --host 0.0.0.0 --port {PORT}" in script
    assert "shutdown" not in user_data("k", "master")


def test_machine_presets_fit_the_model():
    assert "g6.xlarge" in MACHINES and all(price > 0 for _, price, _ in MACHINES.values())
