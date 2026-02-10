# GAIA Service Blueprint: `gaia-web`

## Role and Overview

`gaia-web` is the primary user-facing component of the GAIA system. It is responsible for handling all user interactions, providing a web-based user interface, exposing API endpoints for programmatic access, and integrating with external communication platforms like Discord. Its core function is to translate user requests into `CognitionPacket`s and deliver GAIA's responses back to the users.

## Internal Architecture and Key Components

*   **Entry Point (`gaia_web/main.py`)**:
    *   Initializes the web application framework (FastAPI).
    *   Configures routes for various API endpoints, including `/process_user_input` and `/output_router`.
    *   Handles the startup and shutdown of the Discord bot via `startup_event()` and `shutdown_event()`, interacting with `discord_interface.py`.
    *   Contains the `logger = logging.getLogger("GAIA.Web.API")` instance.
    *   **Note on Logging:** The current logging configuration in `main.py` is implicit, relying on Python's default behavior or `uvicorn`'s settings. This has led to critical `logger.error` messages from `discord_interface.py` not appearing in Docker logs, hindering debugging efforts. (Addressed by recent logging enhancement in candidate).

*   **User Interface (UI)**:
    *   Serves static and dynamic content for the web interface.
    *   Allows users to submit queries, view GAIA's responses, and manage configurations.
    *   Assets are served from the `static/` directory. (Future functionality for web output routing is noted but not yet implemented).

*   **API Endpoints**:
    *   `/process_user_input`: Receives raw user input, constructs an initial `CognitionPacket`, and sends it to `gaia-core`'s `/process_packet` endpoint.
    *   `/output_router`: This is a crucial endpoint called by `gaia-core` *for autonomous messages* or other services when a `CognitionPacket` (containing a response) is ready for delivery *without an originating `gaia-web` user request*. It routes the response to the appropriate destination (e.g., Discord, or logs if no routing info).

*   **`CognitionPacket` Creation and Processing**:
    *   `main.py` is responsible for constructing the initial `CognitionPacket` from user input for `/process_user_input` endpoint.
    *   It receives the fully processed `CognitionPacket` back from `gaia-core` and extracts the final response.

## Discord Integration Details (`discord_interface.py`)

This module (`gaia-web/gaia_web/discord_interface.py`) is dedicated to integrating `gaia-web` with Discord. It manages the Discord bot's lifecycle, message reception, and message sending.

*   **`logger = logging.getLogger("GAIA.Web.Discord")`**: Dedicated logger instance for Discord-related activities.
*   **`DiscordInterface` Class**:
    *   **`__init__(self, bot_token: str, core_endpoint: str)`**: Initializes the bot with the Discord token and the `gaia-core` endpoint.
    *   **`start(self)`**:
        *   Imports `discord.py` and `discord.ext.commands`. Includes an `ImportError` check that logs `"discord.py not installed"` if the library is missing. (Note: This error was observed transiently in logs, but the bot did connect, suggesting it was not a persistent issue.)
        *   Sets up Discord intents and initializes the `commands.Bot`.
        *   Registers `on_ready()` and `on_message()` event handlers.
        *   Calls `bot.start(self.bot_token)` to connect to Discord.
    *   **`on_ready()` event**: Triggered when the bot successfully connects to Discord. Logs connection status and sets the bot's presence.
    *   **`on_message(message)` event**:
        *   Fired when the bot receives any message.
        *   Filters out messages from itself.
        *   Determines if the message is a DM or if the bot was mentioned/addressed in a channel.
        *   If the bot should respond, it cleans the message content (removes mentions) and calls `_handle_message()`.
    *   **`_handle_message(...)`**:
        *   This is the core logic for processing incoming Discord messages.
        *   Constructs an initial `CognitionPacket` from the Discord message content and metadata (channel_id, user_id, etc.). Crucially, it sets `output_routing.primary.destination = OutputDestination.DISCORD`.
        *   Sends this `CognitionPacket` to `gaia-core`'s `/process_packet` endpoint via `httpx.AsyncClient` with a timeout of `600.0` seconds (10 minutes).
        *   **Current Issue:** Despite the long timeout, `gaia-web` is logging `Discord: Request to gaia-core timed out`. This occurs even when `gaia-core` finishes processing within ~20-30 seconds, indicating a premature or misconfigured timeout on `gaia-web`'s side.
        *   Upon receiving a `completed_packet` back from `gaia-core` (or a timeout), it extracts `gaia_response_text = completed_packet.response.candidate`.
        *   Applies `_strip_think_tags()` and `_build_status_footer()` to the response.
        *   Calls `await self._send_response(message_obj, gaia_response_text, is_dm)` to send the response back to Discord.
        *   Includes comprehensive `try...except` blocks for `httpx.TimeoutException`, `httpx.HTTPStatusError`, and a general `Exception` during packet processing, logging errors and sending fallback messages to Discord. The `logger.error("Discord: Request to gaia-core timed out")` is now visible due to improved logging.
    *   **`_send_response(self, message_obj: Any, content: str, is_dm: bool)`**:
        *   Responsible for physically sending the response content to Discord.
        *   Calls `_split_message(content)` to break the response into chunks respecting Discord's 2000-character limit.
        *   Iterates through the message chunks and calls `await message_obj.channel.send(msg)` for each.
        *   **CRITICAL DEBUGGING POINT**: Contains a `try...except Exception as e:` block that logs `logger.error(f"Discord: Failed to send response: {e}")`. Even with enhanced logging, this block does not appear to be logging when the fallback message (after `httpx.TimeoutException`) fails to send. This suggests a deeper logging issue for `GAIA.Web.Discord` or a silent failure within `discord.py` for specific `asyncio` contexts.
    *   **`_split_message(self, content: str, max_length: int = 2000) -> list`**:
        *   Splits long text into a list of strings, attempting to break at newlines or spaces to avoid cutting words, ensuring each chunk is within `max_length`.
*   **Auxiliary Functions**:
    *   **`_strip_think_tags(text: str)`**: Removes internal `<think>` tags and other reflection markers from the LLM's output before it's sent to the user.
    *   **`_build_status_footer(packet: 'CognitionPacket')`**: Creates a compact footer for Discord messages, summarizing model, observer activity, and deployment mode.
    *   **`send_to_channel(...)` and `send_to_user(...)`**: Asynchronous functions used by `main.py`'s `/output_router` (or potentially `_handle_message`) to send messages autonomously or in response to a `CognitionPacket` from `gaia-core`. They also use `_split_message`.
    *   **`start_discord_bot(...)`**: Called by `main.py`'s `startup_event` to initialize and run the Discord bot in a background thread.
    *   **`stop_discord_bot()`**: Called by `main.py`'s `shutdown_event` to gracefully stop the bot.
    *   **`is_bot_ready()`**: Checks if the Discord bot is connected.

## Data Flow and `CognitionPacket` Processing

1.  **User Input (Discord)**: A user sends a message on Discord (DM or mention).
2.  **`on_message()`**: `discord_interface.py`'s bot captures the message.
3.  **`_handle_message()`**: Constructs an initial `CognitionPacket` and dispatches it via HTTP POST to `gaia-core`'s `/process_packet` endpoint.
4.  **`gaia-core` Processing**: `gaia-core` processes the packet, generates a response, and includes `OutputDestination.DISCORD` in the `CognitionPacket`'s `output_routing`. This typically takes **~20-30 seconds** for complex philosophical queries.
5.  **`gaia-core` Returns Packet**: `gaia-core` returns the completed `CognitionPacket` directly to `gaia-web` via an HTTP `200 OK` response.
6.  **`gaia-web` Delivers Response**: `gaia-web`'s `_handle_message()` receives the `completed_packet` and attempts to deliver the response to Discord via `_send_response()`. **However, `gaia-web` often times out waiting for `gaia-core`'s response, leading to `httpx.TimeoutException`.**
7.  **Autonomous Messages (`/output_router`)**: The `/output_router` endpoint in `main.py` is used when `gaia-core` or other services initiate a message *autonomously* (i.e., not in direct response to a `gaia-web` user request).

## Interaction Points with Other Services

*   **`gaia-core`**:
    *   **Caller**: `gaia-web` (via `_handle_message`) sends `CognitionPacket`s to `gaia-core`.
    *   **Callee**: `gaia-web` receives completed `CognitionPacket`s back from `gaia-core`.
*   **Discord API**: Directly interacted with by `discord_interface.py` for message reception and sending.
*   **`gaia-common`**: Utilizes data structures (e.g., `CognitionPacket`) and utility functions defined in `gaia-common`.

## Key Design Patterns within `gaia-web`

*   **API Gateway**: `main.py` acts as the entry point for various interactions.
*   **Asynchronous Communication**: Uses `httpx` for non-blocking calls to `gaia-core` and `asyncio` for Discord bot operations.
*   **Protocol-Oriented Communication**: Heavy reliance on `CognitionPacket` for structured inter-service data exchange.
*   **Event-Driven Architecture**: Discord bot operates on events (`on_message`, `on_ready`).
*   **Message Splitting/Pagination**: Implemented in `_split_message` to adhere to Discord's API limits.

## Key Debugging Challenges

1.  **Premature `httpx.TimeoutException`**: `gaia-web` is experiencing `httpx.TimeoutException` when calling `gaia-core`, despite a configured `httpx.AsyncClient` timeout of `600.0` seconds (10 minutes). `gaia-core` logs indicate it completes processing complex queries within approximately **~20-30 seconds**. This significant discrepancy suggests the timeout is being triggered prematurely.
    *   **Potential Causes**:
        *   **`httpx` Default Inactivity Timeout:** `httpx` has a default 5-second timeout for network *inactivity*. While `timeout=600.0` should override this for the total request, it's possible a sub-timeout (e.g., `connect`, `read`, `write`) is implicitly shorter or being triggered due to a momentary silence during `gaia-core`'s streaming response, or a network hiccup.
        *   **Uvicorn/Docker/Network Timeouts:** Other layers (e.g., Docker's internal networking, `uvicorn`'s own worker timeout, or even an underlying OS TCP timeout) might be silently terminating the connection before `httpx`'s full 10-minute timeout is reached.
        *   **Event Loop Blocking:** A blocking operation within `gaia-web`'s `asyncio` event loop could theoretically starve the `httpx` client, leading to a perceived timeout, though less likely given the direct `await client.post` call.
2.  **Unlogged Fallback Message Failures**: After a `httpx.TimeoutException`, `gaia-web` attempts to send a fallback message to Discord via `_send_response()`. Although `_send_response()` contains a `try...except Exception as e:` block with `logger.error`, these errors are *not* appearing in the Docker logs. This points to a persistent logging configuration issue for the `GAIA.Web.Discord` logger in this specific asynchronous context, or a silent failure within `discord.py` itself that doesn't propagate as a Python exception.
