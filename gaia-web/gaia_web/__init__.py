"""
gaia-web: The Face - UI and API gateway.

This service handles all user-facing interactions:
- HTTP/REST API endpoints
- Server-Sent Events (SSE) streaming
- WebSocket connections
- Static file serving
- Discord/external integrations
- Output routing and formatting

Dependencies:
- gaia-common: Shared protocols and utilities
- gaia-core: Cognitive processing (via HTTP)
"""

__version__ = "0.1.0"
__service__ = "gaia-web"
