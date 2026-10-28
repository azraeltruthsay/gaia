---
name: palace-lookups
description: Remember what you looked up before — facts with their sources and dates
version: 1
execution_mode: PLAYBOOK
legacy_maps_to: palace_lookups
sensitive: false
domain: palace
parameters:
  - name: query
    type: string
    required: false
    description: What to remember (leave empty to browse the most recent lookups)
  - name: top_k
    type: integer
    required: false
    description: Maximum number of results (default 5)
  - name: archive
    type: boolean
    required: false
    description: Also search older, archived lookups
---
Every web search and page fetch you make is remembered, word for word, with
where it came from. This recalls them: each result gives the fact, its source
and url, when you looked it up, and flags — stale (time to check again),
disputed (someone pushed back), answered_well (it answered a question before),
archived (from the older shelf). Cite the source when you use one.
