"""vdaj: a tool call missing a required parameter is a failed result naming it,
not an internal error."""
import asyncio

import pytest

from gaia_mcp import tools


def test_missing_param_names_it():
    with pytest.raises(ValueError, match="needs the parameter 'doc_id'"):
        asyncio.run(tools.execute_limb("cfr_rolling_context", {"target_section": 1}, None,
                                       pre_approved=True))
