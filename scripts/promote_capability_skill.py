#!/usr/bin/env python3
"""
promote_capability_skill.py — Manual promotion of a CodeMind-drafted
capability skill (9ar0) from an inert knowledge/skills/auto/ draft into a
live, callable PLAYBOOK skill.

This is a SEPARATE, human-invoked step — never called from the sleep
cycle or promote_pipeline.sh. CodeMind's drafting path
(sleep_task_scheduler.py's _draft_capability_skill) never writes into
gaia-mcp's live skills directory or flips a draft's status/enabled fields;
only this script does, and only after explicit --approve.

Usage:
    python scripts/promote_capability_skill.py <slug>              # dry preview
    python scripts/promote_capability_skill.py <slug> --approve    # promote

Still requires the normal candidates/ -> production sync/promotion step for
gaia-mcp afterward (see .claude/rules/candidate-first.md) — this script
only prepares the candidate-side artifact, it does not deploy to
production.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "candidates" / "gaia-common"))
sys.path.insert(0, str(PROJECT_ROOT / "candidates" / "gaia-mcp"))

from gaia_common.utils.skill_code_safety import lint_skill_source  # noqa: E402
from gaia_common.utils.skill_sandbox import run_sandboxed  # noqa: E402
from gaia_mcp.skill_package import load_skill_package  # noqa: E402

# The knowledge tree is data, not source — it lives in the sibling
# gaia-instance/ directory on the host and is bind-mounted to the
# absolute path /knowledge inside containers (docker-compose.yml:
# "../gaia-instance/knowledge:/knowledge"). PROJECT_ROOT/knowledge is NOT
# it (verified live during 9ar0: that path exists but is a stray,
# near-empty local dir some containers create, not the real mounted
# tree) — this mirrors the same in-container-vs-host split
# sleep_task_scheduler.py's _draft_capability_skill uses via
# self.config.KNOWLEDGE_DIR.
_KNOWLEDGE_DIR = (
    Path("/knowledge") if Path("/knowledge").is_dir()
    else PROJECT_ROOT.parent / "gaia-instance" / "knowledge"
)
DRAFTS_DIR = _KNOWLEDGE_DIR / "skills" / "auto"
LIVE_SKILLS_DIR = PROJECT_ROOT / "candidates" / "gaia-mcp" / "gaia_mcp" / "skills"


def _fail(msg: str) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(1)


def _frontmatter_field(text: str, key: str) -> str:
    """Best-effort display-only read of one frontmatter field (not the
    parser of record — load_skill_package is, for the fields it knows)."""
    m = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("slug", help="Draft directory name under knowledge/skills/auto/")
    parser.add_argument(
        "--approve", action="store_true",
        help="Actually promote (without this: print a summary and re-validate only).",
    )
    args = parser.parse_args()

    skill_dir = DRAFTS_DIR / args.slug
    skill_md_path = skill_dir / "SKILL.md"
    source_path = skill_dir / "skill_source.py"

    if not skill_md_path.exists() or not source_path.exists():
        _fail(f"draft not found: {skill_dir} (expected SKILL.md + skill_source.py)")

    raw_frontmatter = skill_md_path.read_text(encoding="utf-8")
    pkg = load_skill_package(skill_md_path)
    if pkg is None:
        _fail(f"could not parse frontmatter at {skill_md_path}")

    code = source_path.read_text(encoding="utf-8")

    # Defense in depth: never trust the frontmatter's sandbox_tested flag
    # alone — re-derive syntax validity and safety-lint right now, and
    # require the recorded draft-time sandbox run to have passed too.
    if _frontmatter_field(raw_frontmatter, "sandbox_tested").lower() != "true":
        _fail(
            "SKILL.md records sandbox_tested != true (draft-time sandbox run did not "
            "pass) — will not promote an unsandboxed or failed draft."
        )

    try:
        ast.parse(code)
    except SyntaxError as e:
        _fail(f"skill_source.py fails to parse: {e.msg} (line {e.lineno})")

    lint = lint_skill_source(code)
    if not lint.ok:
        _fail(f"skill_source.py fails safety lint: {lint.violations}")

    # Re-run the sandbox now (cheap confirmation the module still executes
    # cleanly) — a second, independent check on top of the draft-time run.
    sandbox = run_sandboxed(code, {})

    print(f"── Capability skill promotion: {pkg.name} ──")
    print(f"  description:        {pkg.description}")
    print(f"  frontmatter status: status={pkg.status}, enabled={pkg.enabled}")
    print(f"  attempted tool:     {_frontmatter_field(raw_frontmatter, 'attempted_tool_name')}")
    print(f"  source seed:        {_frontmatter_field(raw_frontmatter, 'source_seed')}")
    print(f"  draft-time sandbox: {_frontmatter_field(raw_frontmatter, 'sandbox_summary')}")
    print(f"  source file:        {source_path}")
    print("  re-validated now:   syntax OK, safety lint OK")
    print(f"  sandbox re-check:   ok={sandbox.ok} ({sandbox.error or 'clean'})")

    if not args.approve:
        print("\nDry run only — pass --approve to actually promote this skill.")
        return

    LIVE_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    live_module_path = LIVE_SKILLS_DIR / f"{pkg.name}.py"
    live_module_path.write_text(code, encoding="utf-8")

    # Flip the draft's own SKILL.md to active/enabled — SkillGateway reads
    # frontmatter fresh on each reload, so this is what actually makes the
    # skill routable (Phase A's gate in skill_gateway.py/skill_package.py).
    updated_text = raw_frontmatter.replace("status: draft", "status: active", 1)
    updated_text = updated_text.replace("enabled: false", "enabled: true", 1)
    skill_md_path.write_text(updated_text, encoding="utf-8")

    print(f"\nPROMOTED: {live_module_path}")
    print("SKILL.md flipped to status: active, enabled: true")
    print(
        "\nNOTE: this only prepares the candidate-side artifact. gaia-mcp still needs "
        "its normal candidates/ -> production sync (see .claude/rules/candidate-first.md) "
        "and a container restart/rebuild before the skill is live in production."
    )


if __name__ == "__main__":
    main()
