#!/bin/bash
# Mesh Review — Sequential multi-agent review orchestration
#
# Usage: /mesh-review <target_file_or_directory>
#
# Runs the full specialist agent review chain:
#   1. CodeMind (structural review)
#   2. Sentinel (security review)
#
# Each agent runs in an isolated session with its own context.
# Each produces an independent AgentReviewResult JSON.
# Results are presented sequentially for human review.
#
# Verdict rules:
#   - Any "reject" verdict blocks promotion
#   - "approve_with_notes" requires human acknowledgment
#   - All "approve" = clear to proceed
#   - Human makes the final call — no automated consensus

TARGET="${1:-.}"

echo "=== GAIA Specialist Agent Mesh Review ==="
echo "Target: $TARGET"
echo ""
echo "This will run a sequential multi-agent review:"
echo "  1. CodeMind — structural code review (contracts, conventions, drift)"
echo "  2. Sentinel — security review (boundaries, injection, approval bypass)"
echo ""
echo "Each agent runs independently and produces an AgentReviewResult JSON."
echo "Any 'reject' verdict blocks promotion."
echo ""
echo "--- Step 1: CodeMind Review ---"
echo "Run the CodeMind agent using .claude/agents/codemind/ context against $TARGET."
echo "Wait for AgentReviewResult before proceeding."
echo ""
echo "--- Step 2: Sentinel Review ---"
echo "Run the Sentinel agent using .claude/agents/sentinel/ context against $TARGET."
echo "Wait for AgentReviewResult."
echo ""
echo "--- Step 3: Verdict Summary ---"
echo "Present both AgentReviewResult JSONs side by side."
echo "Summarize: total findings by severity, per-agent verdicts, promotion recommendation."
