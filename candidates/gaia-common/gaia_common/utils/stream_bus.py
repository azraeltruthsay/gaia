"""
Stream Bus Utility (council/cognitive ready)

Central streaming infrastructure for GAIA's spinal column.
- Streams LLM responses chunk-by-chunk (token, sub-token, or fixed char/window)
- Accepts an observer_fn callback for live interruption/interjection
- Supports listener registration for multi-destination streaming
- Integrates with DestinationRegistry for output routing
"""

import logging
from typing import Callable, Optional, List, Any, Generator

logger = logging.getLogger("GAIA.StreamBus")


# --- Listener Registry ---

_output_listeners: List[Callable[[str, Any], None]] = []


def add_output_listener(fn: Callable[[str, Any], None]) -> None:
    """
    Register a listener function that will be called for each streamed token.

    Args:
        fn: Callable that takes (token, **kwargs) and handles the token
    """
    if fn not in _output_listeners:
        _output_listeners.append(fn)
        logger.debug(f"Added output listener: {fn.__name__ if hasattr(fn, '__name__') else fn}")


def remove_output_listener(fn: Callable[[str, Any], None]) -> None:
    """Remove a previously registered listener."""
    if fn in _output_listeners:
        _output_listeners.remove(fn)
        logger.debug(f"Removed output listener: {fn.__name__ if hasattr(fn, '__name__') else fn}")


def clear_output_listeners() -> None:
    """Remove all registered listeners."""
    _output_listeners.clear()
    logger.debug("Cleared all output listeners")


def emit_token(token: str, **kwargs) -> None:
    """
    Emit a token to all registered listeners.

    Args:
        token: The token/chunk to emit
        **kwargs: Additional context (e.g., packet, destination, metadata)
    """
    for fn in _output_listeners:
        try:
            fn(token, **kwargs)
        except Exception as e:
            logger.warning(f"Output listener error: {e}")


# --- Main Streaming Functions ---

def publish_stream(
    response_iterable,
    output_fn: Callable,
    observer_fn: Optional[Callable[[str], str]] = None,
    chunk_size: int = 8,
    flush_every: int = 1,
    emit_to_listeners: bool = True,
    **listener_kwargs
) -> str:
    """
    Streams tokens/chunks from a model response iterator, with observer hooks.

    This is the primary streaming function for the spinal column.

    Args:
        response_iterable: Iterable (string, generator, or LLM stream) of output tokens/chunks
        output_fn: Function to call with each output chunk (e.g., print or web UI update)
        observer_fn: Callback(buffer) -> "continue" or "interrupt" (optional)
        chunk_size: Number of characters per emitted chunk (if input is a string)
        flush_every: Flush output every N chunks (for UI/log responsiveness)
        emit_to_listeners: Whether to emit tokens to registered listeners
        **listener_kwargs: Additional kwargs to pass to listeners (e.g., packet, destination)

    Returns:
        The full streamed response as a string.
    """
    buffer = []
    full_response = ""
    chunk_buffer = ""
    interrupted = False

    if isinstance(response_iterable, str):
        # Convert string to chunked stream
        def chunker(text, n):
            for i in range(0, len(text), n):
                yield text[i:i + n]
        response_iterable = chunker(response_iterable, chunk_size)

    for i, chunk in enumerate(response_iterable):
        if not chunk:
            continue

        chunk_buffer += chunk
        full_response += chunk

        if (i + 1) % flush_every == 0 or "\n" in chunk:
            # Output to primary destination
            try:
                output_fn(chunk_buffer, end="", flush=True)
            except TypeError:
                # Some output functions don't support end/flush kwargs
                output_fn(chunk_buffer)

            # Emit to registered listeners
            if emit_to_listeners and _output_listeners:
                emit_token(chunk_buffer, **listener_kwargs)

            buffer.append(chunk_buffer)
            chunk_buffer = ""

            # Observer check (for interruption/interjection)
            if observer_fn is not None:
                try:
                    decision = observer_fn("".join(buffer))
                    if decision == "interrupt":
                        output_fn("\n[Stream interrupted by observer]\n")
                        logger.warning("Streaming interrupted by observer.")
                        interrupted = True
                        break
                except Exception as e:
                    logger.warning(f"Observer error: {e}")

    # Flush any remaining buffer
    if chunk_buffer:
        try:
            output_fn(chunk_buffer, end="", flush=True)
        except TypeError:
            output_fn(chunk_buffer)

        if emit_to_listeners and _output_listeners:
            emit_token(chunk_buffer, **listener_kwargs)

    if interrupted:
        logger.info("Stream completed (interrupted).")
    else:
        logger.info("Stream completed.")

    return full_response


def create_stream_generator(
    response_iterable,
    observer_fn: Optional[Callable[[str], str]] = None,
    chunk_size: int = 8
) -> Generator[str, None, str]:
    """
    Create a generator that yields tokens and can be passed to DestinationRegistry.

    Args:
        response_iterable: Source of tokens
        observer_fn: Optional observer for interruption
        chunk_size: Chunk size for string sources

    Yields:
        Individual tokens/chunks

    Returns:
        The full response (accessible via generator.send(None) after exhaustion)
    """
    full_response = ""
    buffer = []

    if isinstance(response_iterable, str):
        def chunker(text, n):
            for i in range(0, len(text), n):
                yield text[i:i + n]
        response_iterable = chunker(response_iterable, chunk_size)

    for chunk in response_iterable:
        if not chunk:
            continue

        full_response += chunk
        buffer.append(chunk)

        # Observer check
        if observer_fn is not None:
            try:
                decision = observer_fn("".join(buffer))
                if decision == "interrupt":
                    logger.warning("Stream generator interrupted by observer.")
                    break
            except Exception as e:
                logger.warning(f"Observer error in generator: {e}")

        yield chunk

    return full_response


# --- Integration with Destination Registry ---

def stream_to_destinations(
    response_iterable,
    packet=None,
    observer_fn: Optional[Callable[[str], str]] = None,
    chunk_size: int = 8
) -> str:
    """
    Stream response to destinations via the DestinationRegistry.

    This is the preferred method for streaming in the spinal column architecture.

    Args:
        response_iterable: Source of tokens
        packet: CognitionPacket with routing information
        observer_fn: Optional observer for interruption
        chunk_size: Chunk size for string sources

    Returns:
        The full response as a string
    """
    try:
        from gaia_common.utils.destination_registry import get_registry
        registry = get_registry()

        # Create generator
        gen = create_stream_generator(response_iterable, observer_fn, chunk_size)

        # Collect full response while streaming
        tokens = []
        for token in gen:
            tokens.append(token)

        full_response = "".join(tokens)

        # Route the full response to destinations
        registry.route(full_response, packet)

        return full_response

    except Exception as e:
        logger.error(f"stream_to_destinations failed: {e}")
        # Fallback: just collect the response
        if isinstance(response_iterable, str):
            return response_iterable
        return "".join(response_iterable)
