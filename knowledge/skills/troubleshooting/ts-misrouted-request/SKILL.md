---
name: ts-misrouted-request
description: Troubleshoot a request that took the wrong path — a poem, chat or question handled as a tool or file request, or a lookup's empty result answering in place of the real reply.
version: 3
status: active
execution_mode: PLAYBOOK
domain: troubleshooting
seen_in: [turns, logs]
trigger: "A reply that describes or refuses instead of doing (\"The poem is two lines about…\"), an unexpected 'searched the web' / tool status on a creative or chat turn, or 'Tool routing triggered' logged for an intent that needs no tool."
learned_from: [GAIA_Project-yn31]
escalation_note: "Routing code lives in agent_core.py, a vital organ: propose the fix in candidates/ with a test; promotion needs Azrael's approval."
---
# A request took the wrong path

## Symptoms
- You were asked to make something (poem, haiku, story) and your reply talked
  *about* it instead of making it.
- A turn shows `tool routing fired` for an intent that needs no tool (brainstorming,
  chat, greeting), especially when the selector then says `no tool needed`: the
  routing trigger misread the request, and the turn did a tool-selection pass for
  nothing.
- A turn ran a tool or a web search the request didn't need, or a lookup that found
  nothing was used as the answer (`answered from routed … result` on a request to
  make something).

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
Before you fix anything: `issue(action=file)` with a title, the symptom, the lines
`introspect turns` or `logs` showed you as `evidence` (quote them; don't paraphrase —
a quote that isn't in what you were shown is refused), your suspected cause, and this
playbook's name. It becomes a bead (label `gaia-found`). Add findings with
`issue(action=note)`; move it with `issue(action=status)` (investigating → proposed →
fixed). `issue(action=list)` shows its bead and whether it has been closed.

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
