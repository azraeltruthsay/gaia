# GAIA Self-Reflection Map

This document defines the self-reflection framework for GAIA. Reflection is not an afterthought — it is an internal mechanism of review, improvement, and memory synthesis. This specification outlines when GAIA reflects, what types of outputs it creates, and how those outputs integrate with the structured memory architecture.

## 1. Purpose of Self-Reflection
- Detect behavioral trends or conversation issues.
- Summarize sessions, interactions, or memory deltas.
- Propose improvements to self, codebase, memory, or workflows.
- Prepare curated knowledge for long-term or retrainable memory.

## 2. Triggers for Reflection
- Time-based (e.g. every X minutes of idle time).
- Session-end (after a conversation or task concludes).
- Log analysis (detect errors, anomalies, repetitive issues).
- Memory changes (new documents embedded, facts updated).
- Explicit user prompt ("reflect on that", "what did you learn?").

## 3. Reflection Outputs
- Summary of session or action.
- Noted anomalies or inefficiencies.
- Improvement suggestions (code, behavior, memory).
- Questions for user feedback or clarification.
- Flagged content for Tier 5: Retrainable Memory.

## 4. Storage & Tier Integration
- Outputs are stored in Tier 4: Reflective Memory.
- Critical discoveries may be promoted to Tier 3 (Declarative) or Tier 5 (Retrainable) upon review.
- Reflections may also be vectorized for Tier 2 querying.
- Each reflection artifact is timestamped and indexed by source trigger.

## 5. Reflection Loop Process
1. Detect idle time or trigger condition.
2. Retrieve relevant memory (session, logs, vector hits).
3. Generate reflection summary and observations.
4. Store reflection, optionally notify user.
5. Feed findings into other systems (suggestions folder, retrain queue, etc.).
