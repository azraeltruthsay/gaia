#!/bin/bash
# CodeMind Review — Structural code review agent
#
# Usage: /codemind-review <target_file_or_directory>
#
# Invokes the CodeMind specialist agent to perform structural code review
# against GAIA's architectural contracts, coding conventions, and blueprint
# specifications. Produces an AgentReviewResult JSON.
#
# Orchestration: This is one leg of a multi-agent review. Run independently
# or as part of a sequential review chain (CodeMind → Sentinel).
# Each agent produces an independent verdict. Any "reject" blocks promotion.

TARGET="${1:-.}"

echo "=== CodeMind Structural Review ==="
echo "Target: $TARGET"
echo ""
echo "Review this target against GAIA's architectural contracts and coding conventions."
echo ""
echo "Load your context files from .claude/agents/codemind/context/ and shared references"
echo "from .claude/agents/shared/. Assess on all 5 review dimensions:"
echo "1. Contract compliance"
echo "2. Dependency correctness"
echo "3. Failure mode coverage"
echo "4. Intent alignment"
echo "5. Idiom fidelity"
echo ""
echo "Produce a valid AgentReviewResult JSON as your final output."
echo "See .claude/agents/shared/models/review.py for the schema."
echo "See .claude/agents/codemind/examples/ for calibration examples."
