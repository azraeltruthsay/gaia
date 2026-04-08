# Inspirations

GAIA stands on the shoulders of projects and people whose ideas shaped its architecture. This is not a dependency list -- GAIA does not embed or redistribute their code. These are intellectual debts, acknowledged with respect.

---

## Daniel Miessler -- Fabric

**Project**: [Fabric](https://github.com/danielmiessler/fabric)

Miessler's Fabric framework and his thinking around Personal AI (PAI) demonstrated that AI workflows become dramatically more useful when structured through reusable, composable patterns rather than monolithic prompts. GAIA's pattern-based tool architecture -- where domain-specific prompt patterns are synced, indexed, and invoked as first-class tools -- draws directly from the philosophy Fabric established. The `fabric` tool domain in GAIA's MCP layer and the pattern sync pipeline (`scripts/sync_fabric_patterns.py`) exist because Miessler showed the way.

## Jovovich -- MemPalace

**Project**: [MemPalace](https://github.com/milla-jovovich/mempalace)

MemPalace introduced the idea of spatially organizing AI memory -- rooms, drawers, shelves -- giving structure to what would otherwise be a flat vector store. GAIA's memory palace architecture and the AAAK (Atomic Autonomous Accessible Knowledge) compression dialect both trace their lineage to this project. Several modules in `gaia-common` and `gaia-study` were adapted from MemPalace's approach to knowledge extraction, graph organization, and conversation normalization.

Files with direct adaptation credits:
- `gaia-common/gaia_common/utils/aaak_dialect.py`
- `gaia-common/gaia_common/utils/knowledge_graph.py`
- `gaia-common/gaia_common/utils/convo_normalizer.py`
- `gaia-study/gaia_study/general_extractor.py`

---

## abagames -- slash-criticalthink

**Project**: [slash-criticalthink](https://github.com/abagames/slash-criticalthink)

A slash command that makes AI agents critically self-evaluate their own responses before the user commits to them. Checks for hidden assumptions, logical fallacies, happy-path bias, and hallucination. GAIA uses this as a pre-commit review tool during development sessions.

---

*If your work influenced GAIA and isn't listed here, that's an oversight, not ingratitude. Open an issue.*
