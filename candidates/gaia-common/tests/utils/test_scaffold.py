"""Tests: scaffold library (s4r2) — every shipped template renders from its
own header examples and validates; failures are loud, never half-filled."""

import pytest

from gaia_common.utils.scaffold import (
    ScaffoldError,
    get_scaffold,
    list_scaffolds,
    render,
    scaffold_prompt_block,
)

EXPECTED = {"sleep_task", "fastapi_endpoint", "pytest_module",
            "drift_alert_probe", "restart_manifest"}


def _example_vars(scaffold):
    return {v.name: v.example for v in scaffold.variables}


class TestInventory:
    def test_all_shipped_scaffolds_present(self):
        names = {s.name for s in list_scaffolds()}
        assert EXPECTED <= names

    def test_every_scaffold_has_description_and_vars(self):
        for s in list_scaffolds():
            assert s.description, f"{s.name}: missing description"
            assert s.variables, f"{s.name}: no variables declared"
            for v in s.variables:
                assert v.description, f"{s.name}.{v.name}: missing description"
                assert v.example, f"{s.name}.{v.name}: missing example"

    def test_prompt_block_lists_all(self):
        block = scaffold_prompt_block()
        for name in EXPECTED:
            assert name in block


class TestRendering:
    @pytest.mark.parametrize("name", sorted(EXPECTED))
    def test_renders_and_validates_from_own_examples(self, name):
        """The header examples are the contract: they must produce a valid
        artifact, or the scaffold is teaching the model to fail."""
        s = get_scaffold(name)
        content = render(name, _example_vars(s))
        assert content.strip()
        assert "${" not in content, f"{name}: unfilled placeholder survived"

    def test_unknown_scaffold_raises(self):
        with pytest.raises(ScaffoldError, match="unknown scaffold"):
            render("does_not_exist", {})

    def test_missing_variable_raises(self):
        s = get_scaffold("restart_manifest")
        v = _example_vars(s)
        v.pop("service")
        with pytest.raises(ScaffoldError, match="missing variables"):
            render("restart_manifest", v)

    def test_invalid_python_output_raises(self):
        s = get_scaffold("fastapi_endpoint")
        v = _example_vars(s)
        v["handler_body"] = "return {"  # deliberately broken
        with pytest.raises(ScaffoldError, match="not valid Python"):
            render("fastapi_endpoint", v)

    def test_manifest_renders_as_pending_source_tier(self):
        import json
        s = get_scaffold("restart_manifest")
        m = json.loads(render("restart_manifest", _example_vars(s)))
        assert m["status"] == "pending"
        assert m["tier"] == "source"
        assert m["manifest_version"] == 1
