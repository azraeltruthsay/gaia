---
name: ts-misrouted-request
description: Troubleshoot a request that took the wrong path — a poem, chat or question handled as a tool or file request, or a lookup's empty result answering in place of the real reply.
version: 2
status: active
execution_mode: PLAYBOOK
domain: troubleshooting
trigger: "A reply that describes or refuses instead of doing (\"The poem is two lines about…\"), an unexpected 'searched the web' / tool status on a creative or chat turn, or 'Tool routing triggered' logged for an intent that needs no tool."
learned_from: [GAIA_Project-yn31]
escalation_note: "Routing code lives in agent_core.py, a vital organ: propose the fix in candidates/ with a test; promotion needs Azrael's approval."
---
# A request took the wrong path

## Symptoms
- You were asked to make something (poem, haiku, story) and your reply talked
  *about* it instead.
- Your "What I did" line shows a tool or search the request didn't need.
- In your log, for that turn: `Tool routing triggered for intent: <intent>` where the
  intent needs no tool (brainstorming, chat, greeting), or
  `answering from routed <tool> result (tool-answer shape)` after a lookup that found
  nothing.

## Find it
1. `introspect_logs` service `gaia-core`, search the user's words, to get the turn's
   packet id.
2. Search `Tool routing triggered` and `Intent detection: done` near it. Note the
   intent and whether routing fired.
3. If routing fired: the trigger is `AgentCore._should_use_tool_routing` in
   `gaia_core/cognition/agent_core.py`. Its indicator lists (`file_indicators`,
   `exec_indicators`, `search_indicators`, `web_indicators`, …) are plain substring
   matches against the lowercased message. Find which entry is a substring of the
   message. Common trap: a command or verb that is also an everyday word.
4. If the reply was framed from a tool result: look for
   `answering from routed … (tool-answer shape)` in `PromptBuilder`. A lookup that
   found nothing (`"found": false`) is not an answer to a request to make something.

## Record it
Before you fix anything: `introspect(action=issue_file)` with a title, the symptom, the
log lines you were shown as `evidence` (quote them; don't paraphrase), your suspected
cause, and this playbook's name. It becomes a bead (label `gaia-found`). Add findings
with `issue_note`; move it with `issue_status` (investigating → proposed → fixed).
`introspect(action=issues)` shows its bead and whether it has been closed.

## Fix
- Narrow the indicator so it needs real evidence (e.g. `"write "` only counts beside a
  file signal such as an extension, a path or the word "file"). Don't delete an
  indicator that real tool requests still need.
- Write a regression test with the exact message that misrouted, plus one real tool
  request that must still route.
- Candidate first (`candidates/gaia-core/`), run the candidate tests, replay the turn
  on `gaia-core-candidate`.

## Past cases
- **yn31** (2026-11-08): `"write "` in `file_indicators` sent "Write a two-line poem
  about rain on a tin roof." to `palace(work)`; not-found result answered as the poem.
  Fixed in `5c9495f`.
