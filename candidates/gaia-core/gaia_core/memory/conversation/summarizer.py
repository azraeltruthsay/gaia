# /home/azrael/Project/gaia-assistant/app/memory/conversation/summarizer.py

import logging
from typing import List, Dict, Any
import numpy as np
from gaia_core.config import Config as GAIAConfig, get_config
import requests
import os
from gaia_core.utils import mcp_client

logger = logging.getLogger("GAIA.ConversationSummarizer")


def _get_model_pool():
    """Lazily import and return the shared model_pool singleton.

    Importing `model_pool` at module import time creates a circular
    dependency during application startup. We import it on-demand to
    avoid that.
    """
    try:
        from gaia_core.models.model_pool import get_model_pool as _mp
        return _mp()
    except Exception:
        return None

class ConversationSummarizer:
    """
    Uses the LLM to summarize a conversation history.
    Falls back to placeholder text if no LLM is available.
    """

    def __init__(self, llm=None, embed_model=None): # MODIFIED: Added embed_model
        # Defensive: callers sometimes mistakenly pass a Config instance as the first positional
        # argument (historical bug). If that happens, ignore it and treat llm as unavailable.
        if isinstance(llm, GAIAConfig):
            logger.warning("ConversationSummarizer initialized with Config object as llm; treating llm as None")
            llm = None
        self.llm = llm
        # Prefer an explicitly provided embed_model; otherwise try to obtain one
        # from the shared model_pool. This makes the summarizer robust when
        # callers forget to pass the embedding model.
        try:
            if embed_model:
                self.embed_model = embed_model
            else:
                mp = _get_model_pool()
                self.embed_model = mp.get('embed') if mp is not None else None
        except Exception:
            self.embed_model = embed_model

    def generate_summary(self, messages: List[dict], packet: object = None) -> str:
        if not messages:
            return "(No messages to summarize)"

        try:
            # Determine an LLM to use for summarization. Prefer the injected llm,
            # otherwise try to get the active 'prime' from the shared model_pool.
            # Resolve the LLM to use. Prefer the injected llm; otherwise ask
            # the shared model_pool for a model that satisfies the 'prime' role.
            llm = self.llm
            if not llm:
                try:
                    mp = _get_model_pool()
                    llm = mp.get_model_for_role('prime') if mp is not None else None
                except Exception:
                    llm = None
            if not llm:
                logger.info("ℹ️ No LLM available for summarization. Returning placeholder summary.")
                return "Summary unavailable (LLM not connected)"

            # Allow a brief wait for the embedding model to become available
            # (ModelPool loads embeddings in a background thread). This reduces
            # the chance smart history is disabled due to timing.
            try:
                if not self.embed_model:
                    # wait up to 5 seconds for background embed load to finish
                    mp = _get_model_pool()
                    if mp is not None:
                        self.embed_model = mp.get_embed_model(timeout=5)
            except Exception:
                pass

            # Build a smart history using embeddings when available so summarization
            # focuses on salient past turns rather than the entire history.
            try:
                smart_history = self.build_smart_history(messages, current_input=messages[-1].get('content', ''), max_recent=4, max_salient=3)
            except Exception:
                smart_history = messages

            # Compose a human-friendly conversation excerpt for the LLM
            conversation_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in smart_history)

            # Inject active persona and architecture hints into the prompt when available
            persona = None
            try:
                mp = _get_model_pool()
                persona = mp.get_active_persona() if mp is not None else None
            except Exception:
                persona = None

            persona_desc = ""
            if persona:
                try:
                    persona_desc = f"Active persona: {getattr(persona, 'name', getattr(persona, 'id', 'unknown'))}. "
                except Exception:
                    persona_desc = ""

            # If a packet is provided, include a compact snapshot to ground the
            # summary in the current CognitionPacket. This helps the LLM know
            # what immutable identity and other metadata to reference.
            packet_desc = ""
            try:
                if packet is not None:
                    # Support both dict-like and object packet representations
                    if hasattr(packet, 'header'):
                        hdr = packet.header
                        pid = getattr(hdr.persona, 'identity_id', None) if hdr else None
                        pname = getattr(hdr.persona, 'persona_id', None) if hdr else None
                        packet_desc = f"Packet persona_identity={pid} persona_id={pname}. "
                        # include any short immutable identity fields from content
                        try:
                            data_fields = getattr(packet.content, 'data_fields', [])
                            if data_fields:
                                kvs = []
                                for df in data_fields:
                                    k = getattr(df, 'key', None)
                                    v = getattr(df, 'value', None)
                                    if k and v:
                                        kvs.append(f"{k}={str(v)[:120]}")
                                if kvs:
                                    packet_desc += "Packet content fields: " + ", ".join(kvs) + ". "
                        except Exception:
                            pass
                    elif isinstance(packet, dict):
                        hdr = packet.get('header', {})
                        persona_hdr = hdr.get('persona', {})
                        pid = persona_hdr.get('identity_id')
                        pname = persona_hdr.get('persona_id')
                        packet_desc = f"Packet persona_identity={pid} persona_id={pname}. "
                        try:
                            data_fields = packet.get('content', {}).get('data_fields', [])
                            if data_fields:
                                kvs = []
                                for df in data_fields:
                                    k = df.get('key')
                                    v = df.get('value')
                                    if k and v:
                                        kvs.append(f"{k}={str(v)[:120]}")
                                if kvs:
                                    packet_desc += "Packet content fields: " + ", ".join(kvs) + ". "
                        except Exception:
                            pass
            except Exception:
                packet_desc = ""

            # Attempt to discover MCP endpoint capabilities (best-effort, non-fatal)
            mcp_info = ""
            try:
                endpoint = GAIAConfig().constants.get('MCP_LITE_ENDPOINT') or None
                if endpoint:
                    # try common discovery endpoints
                    for p in ["/capabilities", "/methods", "/jsonrpc"]:
                        try:
                            url = endpoint.replace('/jsonrpc', p)
                            r = requests.get(url, timeout=3)
                            if r.ok:
                                try:
                                    payload = r.json()
                                    mcp_info = f"MCP discovery ({p}): {list(payload.keys()) if isinstance(payload, dict) else str(payload)[:200]}"
                                    break
                                except Exception:
                                    mcp_info = f"MCP discovery ({p}): {r.text[:200]}"
                                    break
                        except Exception:
                            continue
            except Exception:
                mcp_info = ""

            # Best-effort: include the Tier-1 Core Identity summary from the
            # configured identity file (if present) to ground the LLM in GAIA's
            # immutable identity and rules. We parse the JSON and extract the
            # `identity_summary` list when possible to avoid dumping raw JSON.
            core_identity_text = ""
            try:
                cfg = GAIAConfig()
                id_path = getattr(cfg, 'identity_file_path', None) or getattr(cfg, 'IDENTITY_FILE', None)
                if id_path and isinstance(id_path, str) and os.path.exists(id_path):
                    try:
                        import json as _json
                        with open(id_path, 'r', encoding='utf-8') as fh:
                            core_obj = _json.load(fh)
                            if isinstance(core_obj, dict):
                                summary_lines = core_obj.get('identity_summary') or core_obj.get('identity_summary', [])
                                if isinstance(summary_lines, list) and summary_lines:
                                    core_identity_text = ' '.join(str(x) for x in summary_lines[:6])
                                else:
                                    # Fallback: include a short excerpt of the JSON
                                    core_identity_text = _json.dumps(core_obj)[:800]
                    except Exception:
                        core_identity_text = ""
            except Exception:
                core_identity_text = ""

            # Attempt a JSON-RPC discovery call to the MCP endpoint (best-effort).
            try:
                if endpoint:
                    rpc_payload = {"jsonrpc": "2.0", "method": "rpc.discover", "params": {}, "id": "discover"}
                    r = requests.post(endpoint, json=rpc_payload, timeout=3)
                    if r.ok:
                        try:
                            resp = r.json()
                            if isinstance(resp, dict):
                                keys = list(resp.keys())
                                mcp_info = mcp_info or f"MCP rpc.discover: keys={keys}"
                        except Exception:
                            mcp_info = mcp_info or f"MCP rpc.discover: {r.text[:200]}"
            except Exception:
                pass

            prompt = (
                f"You are a concise assistant summarizing a conversation for human operators. {persona_desc} {packet_desc}"
                f"If available, include any relevant MCP capabilities discovered: {mcp_info}\n\n"
                f"Core Identity (excerpt): {core_identity_text}\n\n"
                "Provide a short (3-5 sentence) summary of the following excerpt and list 3 bullet points for action items or important facts:\n\n"
                + conversation_text
            )

            logger.debug("🧠 Requesting LLM-based conversation summary...")
            # Defensive: some callers accidentally pass an embedding model as `llm`.
            # Detect common embedding model types and avoid calling them as LLMs.
            # Defensive: avoid calling an embedding model as an LLM. Detect common
            # embedding types and prefer the model_pool prime when available.
            try:
                # local import to avoid importing heavy torch-backed packages at module import time
                from sentence_transformers import SentenceTransformer
                is_embed_model = isinstance(llm, SentenceTransformer)
            except Exception:
                is_embed_model = False

            if is_embed_model or (hasattr(llm, 'encode') and not hasattr(llm, 'create_chat_completion')):
                logger.warning("ConversationSummarizer: selected llm appears to be an embedding model; attempting to use model_pool prime instead")
                try:
                    mp = _get_model_pool()
                    if mp is not None:
                        alt = mp.get('prime') or mp.get('gpu_prime') or mp.get('cpu_prime')
                        if alt and alt is not llm:
                            llm = alt
                except Exception:
                    pass
            # If after fallback we still don't have an LLM, return a safe placeholder
            if is_embed_model and (not hasattr(llm, 'create_chat_completion') and not callable(llm)):
                return "Summary unavailable (LLM not connected)"

            # Prefer the common create_chat_completion API when present
            # Prefer the create_chat_completion API when present
            if hasattr(llm, 'create_chat_completion'):
                try:
                    result = llm.create_chat_completion(messages=[{'role':'user','content':prompt}], max_tokens=256)
                    raw = result
                except Exception as e:
                    logger.debug("LLM create_chat_completion failed, falling back to other call methods: %s", e)
                    # Try callable LLM first
                    if callable(llm):
                        try:
                            raw = llm(prompt)
                        except Exception:
                            raw = None
                    else:
                        raw = None
            else:
                # If the llm is just an embedder or lacks create_chat, prefer to
                # forward to the pool's prime model via model_pool.forward_to_model
                raw = None
                if callable(llm):
                    try:
                        raw = llm(prompt)
                    except Exception:
                        raw = None
                if raw is None:
                    try:
                        # Forward via model_pool as a last-resort structured call
                        mp = _get_model_pool()
                        if mp is not None:
                            raw = mp.forward_to_model('prime', messages=[{'role': 'user', 'content': prompt}], max_tokens=256)
                        else:
                            raw = None
                    except Exception:
                        raw = None
            # Normalize to a plain string
            if isinstance(raw, dict):
                # Prefer OpenAI-style choices
                choices = raw.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    text = choices[0].get("text")
                    if text is None:
                        text = choices[0].get("message", {}).get("content", "")
                else:
                    # Fallback: serialize full response
                    import json
                    try:
                        text = json.dumps(raw)
                    except Exception:
                        text = str(raw)
            else:
                text = raw
            # Guard against None/empty to avoid noisy stack traces
            if not text:
                return ""
            return text.strip()

        except Exception as e:
            logger.error(f"❌ Failed to summarize conversation: {e}", exc_info=True)
            return "(Error during summarization)"

    def build_smart_history(self, full_history: List[Dict], current_input: str, max_recent: int = 3, max_salient: int = 2) -> List[Dict]: # NEW METHOD
        """
        Builds a context-aware history by combining recent turns with semantically
        relevant turns from the past, using the embedding model.
        """
        if not self.embed_model:
            logger.warning("⚠️ No embedding model available. Smart history is disabled, returning full history.")
            return full_history

        if len(full_history) <= (max_recent * 2):
            return full_history  # Not enough history to need smart selection yet

        # 1. Separate recent history from long-term memory
        recent_history = full_history[-(max_recent * 2):]
        long_term_history = full_history[:-(max_recent * 2)]

        if not long_term_history:
            return full_history

        # 2. Find the most salient turns from long-term memory
        salient_turns_with_similarity = []
        # Group long-term history into pairs of (user, assistant) turns
        # Ensure we only process complete user-assistant pairs for salience
        pairs_to_embed = []
        for i in range(0, len(long_term_history), 2):
            user_turn = long_term_history[i]
            if user_turn.get("role") == "user":
                assistant_turn = long_term_history[i+1] if i+1 < len(long_term_history) and long_term_history[i+1].get("role") == "assistant" else None
                if assistant_turn:
                    pairs_to_embed.append((user_turn, assistant_turn))
                else:
                    # If last user turn without assistant response, include it for embedding
                    pairs_to_embed.append((user_turn, None))

        if not pairs_to_embed:
            return full_history # No complete pairs or single user turn in long-term history

        # local import of util to avoid importing torch at module import time
        try:
            from sentence_transformers import util
        except Exception:
            logger.warning("sentence_transformers.util not available; falling back to lexical heuristics")
            # Fallback: pick most recent turns as salient
            for i, (user_t, assistant_t) in enumerate(pairs_to_embed[:max_salient]):
                salient_turns_with_similarity.append((1.0 - float(i) * 0.01, user_t, assistant_t))
            top_salient_pairs = salient_turns_with_similarity
        else:
            # Encode current input once
            current_embedding = self.embed_model.encode(current_input, convert_to_tensor=True)

            # Encode all long-term turns and calculate similarity
            for user_t, assistant_t in pairs_to_embed:
                combined_text = user_t["content"] + (f" {assistant_t['content']}" if assistant_t else "")
                turn_embedding = self.embed_model.encode(combined_text, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(current_embedding, turn_embedding).item()
                salient_turns_with_similarity.append((similarity, user_t, assistant_t))

        # 3. Sort by relevance and pick the top N
        salient_turns_with_similarity.sort(key=lambda x: x[0], reverse=True)
        top_salient_pairs = salient_turns_with_similarity[:max_salient]

        # 4. Assemble the new history
        smart_history = []
        if top_salient_pairs:
            smart_history.append({"role": "system", "content": "[Recap of relevant past conversation for context]"})
            # Add the turns in their original chronological order
            # Need to map back to original indices to preserve order
            original_indices = {id(turn): i for i, turn in enumerate(full_history)}
            sorted_top_salient_pairs = sorted(top_salient_pairs, key=lambda x: original_indices[id(x[1])])

            for _, user_turn, assistant_turn in sorted_top_salient_pairs:
                smart_history.append(user_turn)
                if assistant_turn:
                    smart_history.append(assistant_turn)
            smart_history.append({"role": "system", "content": "[Resuming recent conversation]"})

        smart_history.extend(recent_history)
        return smart_history
