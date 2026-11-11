from pathlib import Path

import pytest

from dart_marl.envs import RoverEnv, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference_env():
    return RoverEnv(load_config(ROOT / "configs/env/professor_parity.yaml"))
