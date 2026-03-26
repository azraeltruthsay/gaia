# GAIA Brain Region Map

> Butcher-diagram style: sagittal side-view, brain faces LEFT.
> 13 named regions across 3 cognitive tiers.

## Visual Layout

```
SAGITTAL SIDE VIEW (brain faces LEFT)
VP: 280w x 225h, brain image y=20..220

          ┌─ Motor Cortex(4) ─┐
         ╱   (action/tools)    ╲
  ┌─ Prefrontal(1) ─┐  ┌─ Somatosensory(5) ─┐
  │  (executive      │  │  (input parsing)    │
  │   reasoning)     │  └────────┬────────────┘
  ├──────────────────┤    ┌─ Parietal(6) ─┐    ┌─ Occipital(9) ──┐
  │ Orbitofrontal(2) │    │ (spatial/      │    │  (pattern       │
  │ (judgement)      │    │  context)      │    │   recognition)  │
  └──────┬───────────┘    └─────┬──────────┘    └──────┬──────────┘
    ┌─ Broca's(3) ─┐   ┌─ Wernicke's(7) ─┐          │
    │ (language     │   │  (language      │     ┌─ Visual(10) ──┐
    │  output)      │   │   comprehension)│     │  (vision,     │
    └──────┬────────┘   └──────┬──────────┘     │   future)     │
     ┌─ Temporal(8) ──────────────────────┐     └───────────────┘
     │   (memory / semantic retrieval)    │
     └────────────────────────────────────┘
          ┌─ Thalamus(11) ─┐
          │  (relay/triage) │
          └──────┬──────────┘
        ┌─ Cerebellum(12) ─┐
        │  (coordination)   │
        └───────┬───────────┘
          ┌─ Brain Stem(13) ─┐
          │   (reflexes)      │
          └───────────────────┘
```

## Region Table

| # | Region | Tier | GAIA Function | Anatomical Position |
|---|--------|------|---------------|---------------------|
| 1 | Prefrontal | Prime | Executive reasoning, complex planning | Upper-left front |
| 2 | Orbitofrontal | Prime | Value judgement, ethical sentinel | Lower-left front |
| 3 | Broca's Area | Prime | Language generation, response composition | Left-mid, below Sylvian fissure |
| 4 | Motor Cortex | Prime | Action planning, tool execution dispatch | Top center, precentral gyrus |
| 5 | Somatosensory | Core | Input parsing, prompt analysis | Top center, postcentral gyrus |
| 6 | Parietal | Core | Spatial/contextual reasoning, working memory | Upper-mid dome |
| 7 | Wernicke's Area | Core | Language comprehension, intent detection | Mid-left, posterior Sylvian |
| 8 | Temporal | Core | Memory retrieval, semantic search, episodic | Lower band, below Sylvian |
| 9 | Occipital | Core | Pattern recognition, embedding similarity | Back upper |
| 10 | Visual Cortex | Core | Vision processing (future multimodal) | Back lower |
| 11 | Thalamus | Nano | Relay/routing hub, triage classification | Deep center (internal) |
| 12 | Cerebellum | Nano | Coordination, response cleanup, refinement | Lower-right, foliated |
| 13 | Brain Stem | Nano | Reflexes, health checks, heartbeat | Bottom center-right |

## Tier Summary

- **Prime** (4 regions): Prefrontal, Orbitofrontal, Broca's, Motor Cortex — higher cognition
- **Core** (6 regions): Somatosensory, Parietal, Wernicke's, Temporal, Occipital, Visual — operational processing
- **Nano** (3 regions): Thalamus, Cerebellum, Brain Stem — fast reflexes and routing

## Notes

- Coordinates TBD — will be assigned after SAE atlas review determines how many features per region
- Layer ranges per region will map to transformer layer groups from each tier's SAE
- Region edges (start/end anchor curves) follow the existing SVG anatomy paths
- Thalamus is anatomically deep/internal — rendered as a small ellipse behind the Sylvian fissure
