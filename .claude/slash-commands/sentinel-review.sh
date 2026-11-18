#!/bin/bash
# Sentinel Review — Security review agent
#
# Usage: /sentinel-review <target_file_or_directory>
#
# Invokes the Sentinel specialist agent to perform security review against
# GAIA's threat model, trust boundaries, and known vulnerability patterns.
# Produces an AgentReviewResult JSON.
#
# Orchestration: This is one leg of a multi-agent review. Run independently
# or as part of a sequential review chain (CodeMind → Sentinel).
# Each agent produces an independent verdict. Any "reject" blocks promotion.

TARGET="${1:-.}"

echo "=== Sentinel Security Review ==="
echo "Target: $TARGET"
echo ""
echo "Review this target for security vulnerabilities against GAIA's threat model."
echo ""
echo "Load your context files from .claude/agents/sentinel/context/ and shared references"
echo "from .claude/agents/shared/. Focus on:"
echo "- Trust boundary crossings (user input, tool parameters, file access)"
echo "- MCP approval bypass risks"
echo "- Path traversal and shell injection vectors"
echo "- Container boundary violations"
echo "- Prompt injection surface"
echo ""
echo "Produce a valid AgentReviewResult JSON as your final output."
echo "See .claude/agents/shared/models/review.py for the schema."
echo "Metrics must include vuln_count and critical_count."
