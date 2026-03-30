# Decision: The Gateway Principle

**Status:** Active
**Date:** 2026-02 (codified; implicit since initial design)

## Context

GAIA needs to accept input from multiple interfaces: Discord DMs, Discord channels, a web UI, voice input, and potentially future interfaces (Telegram, API clients, etc.). Each interface has different message formats, metadata, and response delivery mechanisms.

## Decision

**gaia-core never knows where a message came from.** All input is normalized into a `CognitionPacket` by gaia-web before reaching the cognitive loop. The packet's `OutputRouting` field specifies where the response should be delivered.

```
Discord ──┐
Web UI  ──┤──→ gaia-web ──→ CognitionPacket ──→ gaia-core
Voice   ──┤                                        │
API     ──┘                                        │
           ┌───────────────────────────────────────┘
           │
           └──→ gaia-web routes response via OutputRouting
                  ├── Discord channel (reply_to_message_id)
                  ├── Discord DM (user_id)
                  ├── Web UI (HTTP response)
                  └── Audio (gaia-audio /synthesize)
```

## Consequences

**Positive:**

- Adding a new interface (e.g., Telegram) requires changes only in gaia-web — zero changes to gaia-core
- The cognitive loop can be tested with synthetic packets without any interface running
- Response routing is declarative and inspectable (it's in the packet)
- Session management is interface-agnostic (session IDs can encode source)

**Negative:**

- gaia-web becomes a single point of failure for all interfaces (mitigated by Docker restart)
- Packet construction is duplicated per-interface in gaia-web (DRY violation, but the packets differ enough that abstraction would be forced)
- Discord-specific features (reactions, embeds, threads) must be translated to/from generic packet fields

## Alternatives Considered

1. **Direct interface → core:** Each interface calls gaia-core directly. Rejected because it would require gaia-core to understand Discord message formats, web session cookies, etc.
2. **Message bus (Redis/RabbitMQ):** Decouple via pub/sub. Rejected as over-engineering for single-instance deployment — adds infrastructure with no practical benefit at current scale.
