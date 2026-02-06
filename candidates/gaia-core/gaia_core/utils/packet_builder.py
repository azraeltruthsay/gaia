"""Utility to build small CognitionPacket snapshots for grounding summarization.

Produces a compact dict snapshot (not a full CognitionPacket instance) which
matches the shape expected by `ConversationSummarizer.generate_summary(..., packet=...)`.
This avoids needing to construct the full dataclass and keeps prompts small.
"""
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Optional
from gaia_core.config import Config as GAIAConfig

CORE_IDENTITY_MAX_CHARS = 800

def build_packet_snapshot(session_id: str, persona_id: str, original_prompt: str, history: List[Dict[str, Any]] = None, mcp_info: Optional[str] = None) -> Dict[str, Any]:
    """Return a minimal dict-shaped packet snapshot suitable for summarization.

    Fields included:
      - header.persona.identity_id (persona_id)
      - header.persona.persona_id
      - header.session_id
      - content.original_prompt
      - content.data_fields: list of {key,value}
      - context.relevant_history_snippet: empty list (optional)
      - _mcp_info: optional debug string
    """
    history = history or []
    cfg = GAIAConfig()
    identity_excerpt = ""
    try:
        id_path = getattr(cfg, 'identity_file_path', None) or getattr(cfg, 'IDENTITY_FILE', None)
        if id_path and isinstance(id_path, str) and os.path.exists(id_path):
            try:
                with open(id_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        summary_lines = data.get('identity_summary') or []
                        if isinstance(summary_lines, list) and summary_lines:
                            identity_excerpt = ' '.join(str(x) for x in summary_lines[:6])[:CORE_IDENTITY_MAX_CHARS]
                        else:
                            identity_excerpt = json.dumps(data)[:CORE_IDENTITY_MAX_CHARS]
            except Exception:
                identity_excerpt = ""
    except Exception:
        identity_excerpt = ""

    packet = {
        'header': {
            'session_id': session_id,
            'persona': {
                'identity_id': persona_id,
                'persona_id': persona_id,
            }
        },
        'content': {
            'original_prompt': original_prompt,
            'data_fields': [
                {'key': 'immutable_identity_excerpt', 'value': identity_excerpt}
            ]
        },
        'context': {
            'relevant_history_snippet': []
        }
    }
    # If caller provided an mcp_info string, include it; otherwise try structured discovery
    if mcp_info:
        packet['content']['data_fields'].append({'key': 'mcp_discovery', 'value': str(mcp_info)[:400]})
    else:
        try:
            from gaia_core.utils import mcp_client
            disc = mcp_client.discover()
            if disc and disc.get('ok') and disc.get('methods'):
                packet['content']['data_fields'].append({'key': 'mcp_methods', 'value': disc.get('methods')})
                try:
                    import logging
                    logger = logging.getLogger('GAIA.PacketBuilder')
                    logger.info(f"Attached MCP methods to packet snapshot: count={len(disc.get('methods'))} endpoint={disc.get('endpoint')}")
                except Exception:
                    pass
            else:
                if disc and disc.get('raw'):
                    packet['content']['data_fields'].append({'key': 'mcp_discovery_raw', 'value': disc.get('raw')})
        except Exception:
            pass

    # include a very small recent history sample for grounding if available
    if history:
        # collapse last up to 6 messages as a simple recent_history field
        recent = history[-6:]
        packet['context']['relevant_history_snippet'] = [{'id': str(i), 'role': m.get('role','user'), 'summary': (m.get('content') or '')[:300]} for i, m in enumerate(recent)]

    return packet
