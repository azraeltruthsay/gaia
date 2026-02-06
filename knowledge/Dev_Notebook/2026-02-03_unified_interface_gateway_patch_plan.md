**Date:** 2026-02-03
**Title:** Detailed Patch Plan: Implementing Unified Interface Gateway

## 1. Goal

To implement the "Unified Interface Gateway" pattern by moving Discord bot functionality into `gaia-web-candidate`, enabling robust routing of messages, and ensuring `gaia-core-candidate` operates as an interface-agnostic processing engine.

## 2. Dependencies and Assumptions

*   **`gaia-core-candidate`**: Will be modified to accept and return `CognitionPacket` objects.
*   **`gaia-web-candidate`**: Will host the Discord bot client, handle inbound request standardization, and outbound response routing.
*   **`gaia-common`**: Expected to contain the `CognitionPacket` definition and related utilities.
*   **`gaia-mcp`**: Remains as the tool execution layer, called by `gaia-core`.
*   **`gaia-study`**: Remains as the background processing layer, called by `gaia-core`.
*   **Existing `gaia.sh` and `test_candidate.sh` scripts**: Assumed to be functional for managing services.

## 3. Phase 1: `gaia-web-candidate` Modifications (Discord Bot & Output Routing)

### 3.1. `gaia-web-candidate/requirements.txt`

*   **Action:** Add necessary Python libraries for Discord interaction.
*   **Change:**
    ```
    # Add to existing requirements.txt
    discord.py[speed]=^2.3.2
    python-dotenv
    httpx # For making async HTTP requests to gaia-core
    ```

### 3.2. `gaia-web-candidate/gaia_web/discord_interface.py` (New File)

*   **Action:** Create a dedicated module for Discord bot logic.
*   **Content (High-Level):**
    ```python
    import os
    import discord
    from discord.ext import commands
    import httpx
    import asyncio
    import json # Import json for CognitionPacket serialization

    # Assuming CognitionPacket can be imported from gaia_common
    # Need to find actual path and structure of CognitionPacket
    from gaia_common.protocols.cognition_packet import CognitionPacket, DataField, PacketType, Metadata # Placeholder

    GAIA_CORE_ENDPOINT = os.getenv("CORE_ENDPOINT", "http://localhost:6415") # Should be configurable

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        print(f"Discord bot connected as {bot.user}")

    @bot.event
    async def on_message(message: discord.Message):
        if message.author == bot.user:
            return

        print(f"Discord message received: {message.content}")

        # 1. Construct CognitionPacket
        packet = CognitionPacket(
            packet_type=PacketType.USER_MESSAGE, # Assuming such an enum exists
            user_message=DataField(value=message.content),
            metadata=Metadata( # Assuming Metadata structure
                source={
                    "type": "discord",
                    "channel_id": str(message.channel.id),
                    "user_id": str(message.author.id),
                    "guild_id": str(message.guild.id) if message.guild else None,
                    "raw_message": message.content # Store original message for context
                }
            )
        )

        # 2. Send packet to gaia-core
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{GAIA_CORE_ENDPOINT}/process_packet", # Define this endpoint in gaia-core
                    json=packet.model_dump_json(), # or .dict() depending on Pydantic version and serialization
                    headers={"Content-Type": "application/json"},
                    timeout=300.0 # Long timeout for LLM processing
                )
                response.raise_for_status()
                # 3. Receive completed packet from gaia-core and extract response
                # Assuming gaia-core directly sends back the completed packet
                completed_packet_dict = response.json()
                completed_packet = CognitionPacket.model_validate(completed_packet_dict) # Or from_dict()
                
                gaia_response = completed_packet.llm_response.value # Assuming this path
                
                # 4. Deliver response back to Discord
                await message.channel.send(gaia_response)

        except httpx.HTTPStatusError as e:
            await message.channel.send(f"Error communicating with GAIA Core: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            await message.channel.send(f"Network error communicating with GAIA Core: {e}")
        except Exception as e:
            await message.channel.send(f"An unexpected error occurred: {e}")


    async def start_bot(token: str):
        await bot.start(token)

    # Output routing function (called by gaia-core for autonomous messages)
    async def route_output_packet(packet: CognitionPacket):
        if packet.metadata and packet.metadata.destination:
            dest = packet.metadata.destination
            if dest.get("type") == "discord":
                channel_id = int(dest["channel_id"])
                channel = bot.get_channel(channel_id)
                if channel:
                    await channel.send(packet.llm_response.value) # Assuming content is here
                else:
                    print(f"Error: Discord channel {channel_id} not found for autonomous message.")
            # Add other destination types here (e.g., web UI, email, etc.)
        else:
            print("Warning: Autonomous packet has no destination metadata.")
    ```

### 3.3. `gaia-web-candidate/gaia_web/main.py`

*   **Action:** Integrate the Discord bot startup and create an output routing endpoint.
*   **Change (High-Level):**
    ```python
    # ... existing imports
    import asyncio
    import os
    from dotenv import load_dotenv # if using .env for Discord token

    from gaia_common.protocols.cognition_packet import CognitionPacket # Import packet
    from .discord_interface import start_bot, route_output_packet # Import discord bot functions

    app = FastAPI(
        title="GAIA Web",
        description="The Face - UI and API gateway",
        version="0.1.0",
    )

    # Load Discord token
    load_dotenv()
    DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

    @app.on_event("startup")
    async def startup_event():
        if DISCORD_BOT_TOKEN:
            print("Starting Discord bot...")
            asyncio.create_task(start_bot(DISCORD_BOT_TOKEN))
        else:
            print("WARNING: DISCORD_BOT_TOKEN not found. Discord bot will not start.")

    # New endpoint for gaia-core to push completed packets or autonomous messages
    @app.post("/output_router")
    async def handle_output_from_core(packet: CognitionPacket):
        await route_output_packet(packet)
        return {"status": "success", "message": "Packet received and routed"}

    # ... existing healthcheck and root endpoints

    # Ensure other endpoints also call gaia-core with CognitionPacket and route output
    @app.post("/process_user_input") # Example existing endpoint for web UI
    async def process_user_input(user_input: str, session_id: str):
        # Construct packet, send to core, receive, route output
        packet = CognitionPacket(
            packet_type=PacketType.USER_MESSAGE,
            user_message=DataField(value=user_input),
            metadata=Metadata(source={"type": "web_ui", "session_id": session_id})
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GAIA_CORE_ENDPOINT}/process_packet",
                json=packet.model_dump_json(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            completed_packet = CognitionPacket.model_validate(response.json())
            # For web_ui, just return the response directly
            return {"response": completed_packet.llm_response.value}
    ```

### 3.4. `gaia-web/Dockerfile` (Candidate Version)

*   **Action:** Ensure the Dockerfile builds the new dependencies.
*   **Change:** Ensure `requirements.txt` is copied and installed *before* `gaia_web` source code. (This should already be handled by existing structure).

## 4. Phase 2: `gaia-core-candidate` Modifications (Packet-based API)

### 4.1. Locate `CognitionPacket` Definition

*   **Action:** Search the codebase for `class CognitionPacket`.
*   **Expected Result:** `gaia-common/protocols/cognition_packet.py` (based on previous observations, but needs verification).

### 4.2. Modify `gaia-core-candidate/gaia_core/main.py`

*   **Action:** Adapt the main API endpoint to accept and return `CognitionPacket` objects.
*   **Change (High-Level):**
    ```python
    # ... existing imports (FastAPI, etc.)
    import httpx
    from gaia_common.protocols.cognition_packet import CognitionPacket # Import packet
    # from .agent_core import AgentCore # Assuming AgentCore processes packets
    # from .config import Config # Assuming configuration is available

    app = FastAPI(
        title="GAIA Core",
        description="The Brain - Cognitive loop and reasoning engine",
        version="0.1.0",
    )

    GAIA_WEB_ENDPOINT = os.getenv("CORE_OUTPUT_ROUTER_ENDPOINT", "http://gaia-web:6414/output_router")

    @app.post("/process_packet")
    async def process_packet_from_gateway(packet: CognitionPacket):
        # 1. Process the incoming packet (using existing AgentCore logic or similar)
        print(f"Core received packet from {packet.metadata.source.get('type')}: {packet.user_message.value}")
        
        # Placeholder for actual processing logic
        processed_response_value = f"Core processed: {packet.user_message.value}"
        
        # 2. Update the packet with the response
        packet.llm_response = DataField(value=processed_response_value)
        packet.packet_type = PacketType.LLM_RESPONSE # Update packet type

        # 3. Send the completed packet back to gaia-web's output router
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    GAIA_WEB_ENDPOINT,
                    json=packet.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                    timeout=300.0
                )
        except httpx.HTTPStatusError as e:
            print(f"Error sending packet to output router: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"Network error sending packet to output router: {e}")

        # Return a simple acknowledgement, as the actual response is routed by web
        return {"status": "processing", "packet_id": packet.packet_id}

    # ... existing healthcheck endpoint
    ```
    *   **Note:** The existing `gaia-core` logic for processing a message (likely in `agent_core.py` or `main.py`) will need to be adapted to accept and return a `CognitionPacket`.

### 4.3. `gaia-core/Dockerfile` (Candidate Version)

*   **Action:** No changes expected unless new dependencies are introduced for packet handling.

## 5. Phase 3: Cleanup and Testing Preparation

### 5.1. `docker-compose.yml` and `docker-compose.candidate.yml`

*   **Action:** Remove the `gaia-discord-bot` service definition from both compose files.
*   **Change:** Delete the entire `gaia-discord-bot` service block.

### 5.2. `gaia_rescue.py`

*   **Action:** Deprecate or remove `gaia_rescue.py` if its functionality is fully absorbed.
*   **Note:** For now, it might remain for legacy or direct testing purposes, but its role as a primary bot runner will be eliminated.

## 6. Testing Strategy

*   **Candidate Build:** Build `gaia-web-candidate` and `gaia-core-candidate` images.
*   **Hybrid Test:**
    1.  Ensure all live services are down.
    2.  Start `gaia-core-candidate` (with GPU).
    3.  Start `gaia-web-candidate`.
    4.  Start `gaia-mcp` (live) and `gaia-study` (live).
    5.  Verify Discord bot functionality: send message, receive response.
    6.  Verify `gaia-core-candidate` can process the `CognitionPacket` and route it back via `gaia-web-candidate`.
    7.  Verify autonomous message routing by triggering a test autonomous message from `gaia-core-candidate`.

## 7. Unknowns/Further Investigation

*   Exact structure and fields of `CognitionPacket` and `DataField` (Pydantic model details).
*   Current mechanism `gaia-core` uses to generate and return responses (will need to adapt to return `CognitionPacket`).
*   Need to determine if `gaia_rescue.py` contains other unique logic not related to Discord.

## 8. Rollback Plan

If issues arise during testing, revert `gaia-web-candidate` and `gaia-core-candidate` to their previous states using Git, and re-implement the separate `gaia-discord-bot` service in `docker-compose.yml` (without GPU).
