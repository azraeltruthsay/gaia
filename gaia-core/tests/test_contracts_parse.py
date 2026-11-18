"""zgs2: four service contracts were invalid YAML for months, so the
SemanticCodex and blueprint tooling silently skipped them."""
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve()
ROOTS = [Path("/gaia/GAIA_Project/contracts/services")] + [
    p / "contracts" / "services" for p in _HERE.parents]
DIR = next((p for p in ROOTS if p.is_dir()), None)


@pytest.mark.skipif(DIR is None, reason="contracts/ not mounted here")
@pytest.mark.parametrize("path", sorted(DIR.glob("*.yaml")) if DIR else [], ids=lambda p: p.name)
def test_contract_is_valid_yaml(path):
    assert isinstance(yaml.safe_load(path.read_text()), dict)
