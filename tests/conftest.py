import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def fixture_item(tmp_path):
    path = Path(__file__).parents[1] / "scripts/create-fixture.py"
    spec = importlib.util.spec_from_file_location("fixture_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_fixture(tmp_path / "input")
