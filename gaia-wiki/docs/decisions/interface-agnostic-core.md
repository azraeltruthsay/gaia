# Decision: Interface-Agnostic Core

**Status:** Active
**Date:** 2026-02

## Context

gaia-core runs the cognitive loop — prompt building, inference, observation, tool execution. It could be tightly coupled to Discord (the primary interface) or designed to be interface-agnostic.

## Decision

**gaia-core has zero knowledge of any specific interface.** It processes `CognitionPacket` objects and returns completed packets. The packet's `OutputRouting` tells gaia-web where to send the response, but gaia-core doesn't act on it.

## Why This Matters

This is a stronger claim than just "we use packets." It means:

1. **No Discord imports in gaia-core** — not even transitive
2. **No interface-specific logic in the cognitive loop** — no "if discord then..." branches
3. **Session IDs encode source** (e.g., `discord_dm_123456789`) but gaia-core treats them as opaque strings
4. **Response formatting is done by gaia-web**, not gaia-core — the cognitive loop returns raw text, and the interface layer handles markdown, embeds, message splitting, etc.

## Consequences

- gaia-core can be tested without any interface running — just POST a packet
- The cognitive loop's behavior is deterministic given the same packet (modulo inference randomness)
- Interface bugs never break cognition, and cognition bugs never break interfaces
- Adding a new interface is a gaia-web change, not a gaia-core change

## Related

- [Gateway Principle](gateway-principle.md) — the companion decision about gaia-web's role
- [Cognition Packets](../systems/cognition-packets.md) — the packet format that enables this
