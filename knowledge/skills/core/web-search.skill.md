---
name: web-search
description: Search the web for current information, news, facts, and documentation
version: 1
execution_mode: PLAYBOOK
legacy_maps_to: web_search
sensitive: false
domain: web
parameters:
  - name: query
    type: string
    required: true
    description: The search query
  - name: max_results
    type: integer
    required: false
    description: Maximum number of results (default 3)
---
Search the web using the configured search API (Serper/SearXNG).
Returns structured results with title, snippet, and URL.
Use for: current events, prices, documentation, factual verification.
