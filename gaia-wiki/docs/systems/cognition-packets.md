# Cognition Packets

The `CognitionPacket` is the universal message format for all communication between gaia-web and gaia-core.

## Structure

```python
CognitionPacket:
  version: str          # Schema version
  header: Header        # Routing, session, persona, origin
  intent: Intent        # User intent classification
  context: Context      # Session history, constraints
  content: Content      # Original prompt + data fields
  reasoning: Reasoning  # Chain-of-thought (filled by core)
  response: Response    # Generated response (filled by core)
  governance: Governance # Safety flags, dry run mode
  metrics: Metrics      # Token usage, latency
  status: Status        # Processing state, next steps
  tool_routing: ToolRoutingState  # MCP tool call tracking
```

## Lifecycle

1. **INITIALIZED** — packet created by gaia-web with user input
2. **PROCESSING** — gaia-core is running the cognitive loop
3. **COMPLETED** — response generated, hashes computed
4. Returned to gaia-web for output routing

## Key Fields

### OutputRouting

Specifies where the response should be delivered:

```python
OutputRouting:
  primary: DestinationTarget  # Where to send the response
    destination: "discord" | "web" | "audio" | "log"
    channel_id: str           # Discord channel
    user_id: str              # Discord user (for DMs)
    reply_to_message_id: str  # Thread/reply context
  source_destination: str     # Where the input came from
  addressed_to_gaia: bool     # Was GAIA mentioned/DMed?
```

### Hashing

Packets compute content hashes for integrity verification:

- `content_hash` — hash of the original prompt
- `response_hash` — hash of the generated response

## Location

Defined in `gaia-common/gaia_common/protocols/cognition_packet.py` — shared across all services.
