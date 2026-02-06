**Date:** 2026-02-03
**Title:** Continuation Plan: Implementing CognitionPacket-based Unified Interface Gateway

## 1. Current Status Review

The previous steps successfully established Discord bot integration within `gaia-web-candidate` and set up an `/output_router` endpoint. However, the inter-service communication between `gaia-web` and `gaia-core` is currently using a simplified JSON payload (`MessageRequest`/`MessageResponse` and `OutputRouteRequest`), rather than the full `CognitionPacket` as envisioned in the architectural proposal.

The `gaia-core`'s processing logic (`_process_with_agent_core`) is also a placeholder, not yet integrated with `AgentCore` using `CognitionPacket`s.

## 2. Re-affirming the Goal: CognitionPacket as the Universal Envelope

The core goal of the Unified Interface Gateway is to use the `CognitionPacket` as the single, standardized envelope for all communication between `gaia-web` (the interface layer) and `gaia-core` (the cognitive engine). This enables:
-   Rich metadata tracking (source, destination, session, persona, etc.).
-   Robust routing capabilities for both user-initiated and autonomously generated messages.
-   A clean separation of concerns, making `gaia-core` truly interface-agnostic.

## 3. Detailed Continuation Plan

The following steps outline the modifications required to fully implement the `CognitionPacket`-based Unified Interface Gateway.

### Phase 1: `gaia-web-candidate` - Full CognitionPacket Integration

**3.1. Update `discord_interface.py`**

*   **Objective:** Modify Discord bot logic to construct and process full `CognitionPacket` objects.
*   **Actions:**
    *   Import `CognitionPacket`, `Header`, `Persona`, `Origin`, `OutputRouting`, `DestinationTarget`, `Content`, `DataField`, etc. from `gaia_common.protocols.cognition_packet`.
    *   In `DiscordInterface._handle_message`:
        *   Replace the `payload` dictionary with a properly constructed `CognitionPacket` instance.
        *   Populate `packet.header.origin` with `Origin.USER`.
        *   Populate `packet.header.output_routing.primary` with a `DestinationTarget` of `OutputDestination.DISCORD` and relevant Discord channel/user IDs.
        *   Populate `packet.content.original_prompt` and create a `DataField` for the user message.
        *   Send `packet.model_dump_json()` (or equivalent serialization) to `gaia-core`'s *new* `/process_packet` endpoint.
        *   Expect a `CognitionPacket` in return from `gaia-core`.
        *   Extract `packet.response.candidate` to send back to Discord.
    *   In `send_to_channel` / `send_to_user` (for autonomous messages): These functions will be called by `gaia-web`'s `output_router` when it receives an autonomous `CognitionPacket` from `gaia-core`. They will extract the response from the incoming packet and send it to Discord.

**3.2. Update `gaia_web/main.py`**

*   **Objective:** Modify the API to accept `CognitionPacket`s for routing and processing, and integrate `CognitionPacket` in any direct interaction with `gaia-core`.
*   **Actions:**
    *   Import `CognitionPacket` from `gaia_common.protocols.cognition_packet`.
    *   Remove `OutputRouteRequest` and `OutputRouteResponse` Pydantic models.
    *   Modify the `/output_router` endpoint to accept an incoming `CognitionPacket` directly in the request body.
    *   Within `/output_router`, use `packet.header.output_routing.primary` (or `destination` field) to determine the correct target and call `discord_interface.send_to_channel`/`send_to_user` with the `packet.response.candidate`.
    *   Any other existing endpoints that directly interact with `gaia-core` (e.g., `/process_user_input` for a web UI) should also be updated to construct and send `CognitionPacket`s.

### Phase 2: `gaia-core-candidate` - Full CognitionPacket API

**3.3. Update `gaia_core/main.py`**

*   **Objective:** Adapt `gaia-core`'s main processing endpoint to accept and return `CognitionPacket` objects.
*   **Actions:**
    *   Import `CognitionPacket` from `gaia_common.protocols.cognition_packet`.
    *   Remove `MessageRequest` and `MessageResponse` Pydantic models.
    *   Rename the `/process_message` endpoint to `/process_packet`.
    *   Modify the `/process_packet` endpoint to accept an incoming `CognitionPacket` in the request body.
    *   **Crucially:** The placeholder function `_process_with_agent_core` (or the actual `AgentCore` call when integrated) must be updated to accept and return a `CognitionPacket`.
    *   After `AgentCore` processing, `gaia-core` must send the *completed* `CognitionPacket` via HTTP POST to `gaia-web`'s `/output_router` endpoint.

### Phase 3: `gaia-core-candidate` - AgentCore Integration

**3.4. Update `AgentCore` and Related Cognitive Modules**

*   **Objective:** Ensure the core cognitive loop (`AgentCore.run_turn()`) is designed to operate solely on `CognitionPacket` objects, reading input from the packet and writing results back into the packet.
*   **Actions:**
    *   Identify the entry point for `AgentCore`'s processing.
    *   Modify `AgentCore.run_turn()` to take an incoming `CognitionPacket` as its primary input.
    *   Throughout the cognitive process, all internal data handling, tool calls, and response generation should read from and write back into the fields of this `CognitionPacket`.
    *   The `AgentCore.run_turn()` method should ultimately return a fully populated, completed `CognitionPacket`.

## 4. Testing Strategy (Hybrid Workflow)

After implementing changes for Phase 1 and 2:
1.  **Stop all containers:** `./gaia.sh live stop && ./gaia.sh candidate stop`
2.  **Build Candidates:** Rebuild `gaia-web-candidate` and `gaia-core-candidate` to incorporate the new code.
3.  **Start Mixed Stack:**
    *   Start `gaia-mcp` (live) and `gaia-study` (live).
    *   Start `gaia-core-candidate` (with GPU resources, if applicable).
    *   Start `gaia-web-candidate`.
4.  **Functional Test:**
    *   Send a message to the Discord bot.
    *   Verify `gaia-web-candidate` sends a `CognitionPacket` to `gaia-core-candidate`.
    *   Verify `gaia-core-candidate` processes it and sends a `CognitionPacket` back to `gaia-web-candidate`'s `/output_router`.
    *   Verify `gaia-web-candidate` routes the response back to Discord.
    *   (Future) Implement a test for an autonomous message from `gaia-core-candidate` to `gaia-web-candidate` for routing to Discord.

## 5. Promotion and Finalization

Once hybrid testing is successful, the `gaia-web` and `gaia-core` candidates will be promoted to live. A final dev journal entry will document the successful implementation and status.

---

This plan outlines a clear path forward to fully realizing the "Unified Interface Gateway" architecture.
