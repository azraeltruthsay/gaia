**Date:** 2026-02-03
**Title:** Architectural Proposal: Unified Interface Gateway

## 1. Objective

To refactor GAIA's communication architecture to cleanly support multiple input/output channels (e.g., Discord, Web UI, etc.) and enable autonomous, GAIA-initiated messages. The current model of running interface-specific listeners within or alongside `gaia-core` creates resource contention and lacks a clear, scalable routing strategy.

## 2. Proposed Architecture: The Unified Interface Gateway

This proposal designates the `gaia-web` service as the **Unified Interface Gateway**, making it the single entry and exit point for all external communications. It will be responsible for both receiving requests and routing final responses.

### 2.1. Key Responsibilities of `gaia-web`

-   **Interface Management:** Host the logic for specific communication protocols (e.g., the Discord bot client, WebSocket servers for a web UI).
-   **Inbound Standardization:** Receive requests from external sources and translate them into a standardized `CognitionPacket`. This includes adding rich `source` metadata to the packet (e.g., `{ "type": "discord", "channel_id": "..." }`).
-   **Request Forwarding:** Send the standardized `CognitionPacket` to the `gaia-core` service for processing.
-   **Output Routing:** Receive the completed `CognitionPacket` back from `gaia-core`. Inspect the packet's `source` (for replies) or `destination` (for autonomous messages) metadata to determine the correct output channel.
-   **Response Delivery:** Use the appropriate client (e.g., Discord client) to deliver the final message to the end-user or target channel.

### 2.2. Role of `gaia-core`

`gaia-core` becomes a pure, interface-agnostic "brain". It's only responsibility is to process a `CognitionPacket` and return a completed `CognitionPacket`. It will not contain any code related to specific communication channels like Discord or WebSockets.

## 3. How It Works: The Flow of Information

### 3.1. User-Initiated Request (e.g., Discord Message)

1.  A user sends a message in a Discord channel.
2.  The Discord bot listener *within `gaia-web`* receives the event.
3.  `gaia-web` constructs a `CognitionPacket` containing the message content and adds `source` metadata: `{ "type": "discord", "channel_id": "12345", "user_id": "67890" }`.
4.  `gaia-web` sends the packet to `gaia-core` via an HTTP request.
5.  `gaia-core` processes the packet and generates a response.
6.  `gaia-core` sends the *completed* packet back to an endpoint on `gaia-web`.
7.  `gaia-web` inspects the packet, sees the `source` is Discord, and uses its Discord client to send the response to channel "12345".

### 3.2. GAIA-Initiated Statement

1.  `gaia-core`'s internal initiative loop generates a thought.
2.  `gaia-core` constructs a `CognitionPacket`, populating a `destination` field: `{ "type": "discord", "channel_id": "announcements" }`.
3.  `gaia-core` sends this packet to the same endpoint on `gaia-web`.
4.  `gaia-web` inspects the packet, sees the `destination`, and uses its Discord client to post the message to the "announcements" channel.

## 4. Implementation Plan (High-Level)

1.  **Write a detailed patch plan.**
2.  **Apply changes to `candidate` containers first.** This will primarily involve modifying `gaia-web-candidate` and `gaia-core-candidate`.
3.  **Conduct hybrid testing** to ensure the new `gaia-web-candidate` communicates correctly with the live `gaia-core`.
4.  **Promote** the candidate code to the live environment, carefully managing pathing in Dockerfiles.
5.  **Write a concluding dev journal** to summarize the results.
