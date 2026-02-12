"""
Intent Detection (pillar-compliant, robust)
- Fast reflex/regex path for autonomic commands (help, exit, shell, etc).
- LLM-powered detection for all ambiguous/natural input.
- Returns simple intent labels for downstream pipeline use.
- Ready for expansion: streaming, multi-intent, or advanced \u201cplan\u201d logic.
"""

import re
import logging
from dataclasses import dataclass
import sys
from typing import Optional

logger = logging.getLogger("GAIA.IntentDetection")

@dataclass
class Plan:
    intent: str
    read_only: bool = False

# ---- Reflex path for \u201cautonomic\u201d commands ----
def fast_intent_check(text):
    text = text.lower().strip()
    # Reflexes\u2014no model call
    if text in {"exit", "quit", "bye"}:
        return "exit"
    if text.startswith("help") or text in {"", "h"}:
        return "help"
    if text.startswith("ls ") or text.startswith("cat ") or text.startswith("pwd"):
        return "shell"
    if _detect_read_file_request(text):
        return "read_file"
    # Fast path for explicit find/locate requests
    if ("find" in text or "locate" in text or "search" in text) and ("dev_matrix" in text or "dev matrix" in text or "file" in text):
        return "find_file"
    # Add more hardcoded patterns as needed
    return None

# ---- Model-powered intent detection ----
def _detect_direct_list_tools(text: str) -> bool:
    lowered = (text or "").lower()
    return ("mcp" in lowered and ("tool" in lowered or "action" in lowered)) or "list tools" in lowered

def _detect_tree_request(text: str) -> bool:
    lowered = (text or "").lower()
    tree_keywords = [
        "directory tree",
        "tree of your architecture",
        "folder tree",
        "ls -r",
        "ls -R",
        "read a tree",
        "list your directory",
        "list directory contents",
        "list the directory",
        "list your filesystem",
        "list file system",
        "list your files",
        "directory contents",
        "tree of gaia-assistant",
        "list the tree",
    ]
    if any(k in lowered for k in tree_keywords):
        return True
    # keyword combo: directory + list/contents
    return ("directory" in lowered or "filesystem" in lowered or "file system" in lowered) and ("list" in lowered or "contents" in lowered or "tree" in lowered)

def _detect_list_files_request(text: str) -> bool:
    lowered = (text or "").lower()
    list_files_keywords = [
        "list files",
        "list all files",
        "show me the files",
        "ls -l",
        "ls -la",
    ]
    if any(k in lowered for k in list_files_keywords):
        return True
    return ("list" in lowered and "files" in lowered)

def _detect_read_file_request(text: str) -> bool:
    lowered = (text or "").lower()
    read_keywords = [
        "read ",
        "open ",
        "show ",
        "view ",
        "display ",
        "cat ",
    ]
    file_markers = [
        ".md",
        ".txt",
        ".json",
        ".py",
        ".yaml",
        ".yml",
        ".sh",
        "/",
    ]
    if any(k in lowered for k in read_keywords) and any(m in lowered for m in file_markers):
        return True
    # Explicit filename mention without extension but with read intent
    if any(k in lowered for k in read_keywords) and "file" in lowered:
        return True
    return False
def _mentions_file_like_action(text: str) -> bool:
    """
    Lightweight heuristic to determine whether the user explicitly mentioned
    reading/writing files, logs, or paths.  Intent models occasionally emit
    read/write intents even for normal conversational prompts; this guard
    prevents those false positives from short-circuiting the chat flow.
    """
    lowered = (text or "").lower()
    keywords = [
        "file",
        "log",
        "cat ",
        "open ",
        "read ",
        "write",
        "save",
        "path",
        ".txt",
        ".md",
        "/",
    ]
    return any(token in lowered for token in keywords)


def _detect_fragmentation_request(text: str) -> bool:
    """
    NLU-based detection for requests that need fragmented generation.

    Identifies recitation, long-form content, and known work requests that
    typically exceed token limits and benefit from fragmentation mode.

    Returns True if the request should trigger fragmentation.
    """
    lowered = (text or "").lower()

    # Recitation action verbs - explicit requests to perform/recite
    recitation_verbs = [
        "recite",
        "perform",
        "declaim",
        "read aloud",
        "speak ",
        "tell me ",
        "give me ",
        "say ",
    ]

    # Long-form modifiers that indicate wanting complete content
    long_form_modifiers = [
        "full ",
        "entire ",
        "complete ",
        "whole ",
        "all of ",
        "in full",
        "from start to finish",
        "beginning to end",
        "word for word",
        "in its entirety",
    ]

    # Content types that often need fragmentation
    work_types = [
        "poem",
        "song",
        "story",
        "tale",
        "passage",
        "monologue",
        "soliloquy",
        "speech",
        "verse",
        "stanza",
        "lyrics",
        "ballad",
        "sonnet",
        "epic",
        "fable",
        "nursery rhyme",
        "constitution",
        "declaration",
        "manifest",
        "protocol",
        "charter",
        "document",
    ]

    # Well-known works that are likely to be lengthy
    known_works = [
        "the raven",
        "jabberwocky",
        "ozymandias",
        "the road not taken",
        "invictus",
        "howl",
        "if—",
        "if by rudyard",
        "do not go gentle",
        "stopping by woods",
        "the wasteland",
        "kubla khan",
        "rime of the ancient mariner",
        "the tyger",
        "to be or not to be",
        "hamlet",
        "romeo and juliet",
        "macbeth",
        "casey at the bat",
        "paul revere",
        "charge of the light brigade",
        "gunga din",
        "annabel lee",
        "the bells",
        # GAIA core documents
        "gaia constitution",
        "the constitution",
        "declaration of artisanal",
        "artisanal intelligence",
        "layered identity",
        "identity model",
        "coalition of minds",
        "mindscape manifest",
        "cognition protocol",
    ]

    # Check for explicit recitation verbs
    has_recitation_verb = any(v in lowered for v in recitation_verbs)

    # Check for long-form modifiers
    has_long_form = any(m in lowered for m in long_form_modifiers)

    # Check for work types
    has_work_type = any(w in lowered for w in work_types)

    # Check for known works
    has_known_work = any(w in lowered for w in known_works)

    # Decision logic:
    # 1. Explicit recitation verb + work type = fragmentation
    # 2. Long-form modifier + work type = fragmentation
    # 3. Known work (regardless of verb) = fragmentation
    # 4. Recitation verb + known work = fragmentation

    if has_recitation_verb and has_work_type:
        logger.debug("Fragmentation detected: recitation verb + work type")
        return True

    if has_long_form and has_work_type:
        logger.debug("Fragmentation detected: long-form modifier + work type")
        return True

    if has_known_work:
        logger.debug("Fragmentation detected: known work reference")
        return True

    # Additional pattern: "recite [title]" without explicit work type
    if has_recitation_verb:
        # Match quoted strings or capitalized sequences after recite verbs
        title_pattern = r'recite\s+["\']?([A-Z][^"\'\.]+)["\']?'
        if re.search(title_pattern, text, re.IGNORECASE):
            logger.debug("Fragmentation detected: recite + title pattern")
            return True

    return False


def _detect_tool_routing_request(text: str) -> bool:
    """
    NLU-based detection for requests that need MCP tool routing.

    Identifies explicit tool usage requests that should be routed through
    the GCP Tool Routing System for structured selection and confidence review.

    Returns True if the request should trigger tool routing.
    """
    lowered = (text or "").lower()

    # Explicit MCP/tool action verbs
    tool_action_verbs = [
        "use mcp",
        "call mcp",
        "invoke ",
        "run tool",
        "execute tool",
        "use the tool",
        "call the tool",
    ]

    # Strong file operation patterns (more specific than read_file intent)
    strong_file_patterns = [
        r"read\s+(?:the\s+)?(?:contents?\s+of\s+)?['\"]?/",  # read /path or read '/path'
        r"open\s+(?:the\s+)?file\s+['\"]?/",                 # open the file /path
        r"show\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+of\s+)?['\"]?/",
        r"cat\s+['\"]?/",                                     # cat /path
        r"write\s+to\s+['\"]?/",                              # write to /path
        r"save\s+(?:to|as)\s+['\"]?/",                        # save to /path
    ]

    # Command execution patterns
    exec_patterns = [
        r"run\s+(?:the\s+)?(?:command|script)\s+",
        r"execute\s+(?:the\s+)?(?:command|script)\s+",
        r"shell\s+command\s+",
    ]

    # Check explicit tool verbs
    for verb in tool_action_verbs:
        if verb in lowered:
            logger.debug(f"Tool routing detected: explicit verb '{verb}'")
            return True

    # Check strong file patterns
    for pattern in strong_file_patterns:
        if re.search(pattern, lowered):
            logger.debug(f"Tool routing detected: file pattern")
            return True

    # Check execution patterns
    for pattern in exec_patterns:
        if re.search(pattern, lowered):
            logger.debug(f"Tool routing detected: exec pattern")
            return True

    return False


def _fast_track_intent_detection(text: str) -> Optional[str]:
    """
    Fast-track intent detection for common conversational patterns.
    """
    lowered = (text or "").lower()

    # Feedback patterns
    feedback_keywords = ["feedback", "suggestion", "idea", "improvement"]
    if any(keyword in lowered for keyword in feedback_keywords):
        return "feedback"

    # Brainstorming patterns
    brainstorming_keywords = ["brainstorm", "ideas for", "what if", "how about"]
    if any(keyword in lowered for keyword in brainstorming_keywords):
        return "brainstorming"

    # Correction patterns
    correction_keywords = ["you're wrong", "that's not right", "actually,", "in fact,"]
    if any(keyword in lowered for keyword in correction_keywords):
        return "correction"

    # Clarification patterns
    clarification_keywords = ["what do you mean", "can you explain", "tell me more", "in other words"]
    if any(keyword in lowered for keyword in clarification_keywords):
        return "clarification"

    return None

def model_intent_detection(text, config, lite_llm=None, full_llm=None, fallback_llm=None, probe_context=""):
    """
    Uses an LLM (Lite if present, else Prime) to detect intent for natural language input.
    Output should always be one of: read_file, write_file, mark_task_complete, reflect, seed, shell, list_tools, list_tree, tool_routing, other.

    Args:
        probe_context: Optional domain hint from semantic probe (e.g.
            "Domain context: user references dnd_campaign entities (Rogue's End, Tower Faction)")
    """
    # Fast-track conversational intents
    fast_track_intent = _fast_track_intent_detection(text)
    if fast_track_intent:
        logger.info(f"Fast-track intent detection: {fast_track_intent}")
        return fast_track_intent

    # Prefer Lite for intent detection (lightweight, less failure-prone). Avoid
    # Prime/vLLM here to reduce CUDA/multiprocessing errors during routing. If
    # Lite is unavailable, fall back to "other" rather than touching heavier models.
    def _is_llama_cpp_instance(m):
        try:
            mod = getattr(m.__class__, "__module__", "") or getattr(m, "__module__", "")
            return isinstance(mod, str) and mod.startswith("llama_cpp")
        except Exception:
            return False

    candidates = [
        ("lite", lite_llm),
    ]
    model = None
    for label, cand in candidates:
        if cand is None:
            continue
        model = cand
        logger.debug("Using %s_llm for intent detection.", label)
        break
    if model is None:
        logger.warning("No lite model available for intent detection; falling back to 'other'.")
        return "other"

    # NLU-based fragmentation detection - check BEFORE model-specific paths
    # This ensures consistent detection regardless of backend (llama_cpp or LLM)
    if _detect_fragmentation_request(text):
        logger.info("NLU fragmentation detection: recitation intent detected")
        return "recitation"

    # NLU-based tool routing detection - check for explicit MCP tool requests
    if _detect_tool_routing_request(text):
        logger.info("NLU tool routing detection: tool_routing intent detected")
        return "tool_routing"

    # Hard override: user clearly asked to list MCP tools/actions.
    # if _detect_direct_list_tools(text):
    #     return "list_tools"
    # if _detect_tree_request(text):
    #     return "list_tree"
    # if _detect_list_files_request(text):
    #     return "list_files"
    # if _detect_read_file_request(text):
    #     return "read_file"
    # Simple file-discovery heuristic: if user asks to find/locate a file (e.g., dev_matrix), route to find_file intent.
    lowered = (text or "").lower()
    if "find" in lowered or "locate" in lowered or "search" in lowered:
        if "file" in lowered or "dev_matrix" in lowered or ".md" in lowered or ".json" in lowered:
            return "find_file"
    if "dev_matrix" in lowered:
        return "find_file"

    # Defensive shortcut: if the selected model is backed by llama_cpp, avoid
    # calling its create_chat_completion and use a lightweight heuristic.
    if _is_llama_cpp_instance(model):
        logger.warning("Detected llama_cpp backend for intent detection; using heuristic fallback.")
        # Very small heuristic: map leading verb/keyword to intent when obvious.
        first = (text or "").strip().split()
        first_word = first[0].lower() if first else ""
        mapping = {
            "read": "read_file",
            "open": "read_file",
            "cat": "read_file",
            "show": "read_file",
            "write": "write_file",
            "save": "write_file",
            "complete": "mark_task_complete",
            "done": "mark_task_complete",
            "help": "other",  # reflex handles help; here treat safely
            "exit": "other",
            "quit": "other",
            "shell": "shell",
            # Recitation/fragmentation verbs (backup - NLU check above should catch most)
            "recite": "recitation",
            "perform": "recitation",
            "declaim": "recitation",
            "tell": "recitation",  # "tell me the raven"
            "sing": "recitation",  # "sing me a song"
            "clarify": "clarification",
            "explain": "clarification",
            "correct": "correction",
            "actually": "correction",
            "brainstorm": "brainstorming",
            "feedback": "feedback",
            "chat": "chat",
        }
        intent_guess = mapping.get(first_word, "other")
        if intent_guess in {"read_file", "write_file"} and not _mentions_file_like_action(text):
            return "other"
        return intent_guess
    # Build context line from semantic probe (if available)
    context_line = ""
    if probe_context:
        context_line = f"Context: {probe_context}\n"

    prompt = (
        "You are an intent detector for GAIA. Return exactly one intent from:\n"
        "read_file, write_file, mark_task_complete, reflect, seed, shell, list_tools, list_tree, list_files, "
        "recitation (the user wants to recite a known work like a poem, song, or story), "
        "clarification (the user is asking for more information or clarification), "
        "correction (the user is correcting a previous statement by GAIA), "
        "brainstorming (the user is in a creative or brainstorming mode), "
        "feedback (the user is providing feedback on GAIA's performance), "
        "chat (a general conversational intent), "
        "other.\n"
        f"{context_line}"
        f"User: {text}\nIntent:"
    )
    messages = [
        {"role": "system", "content": "Intent detection agent."},
        {"role": "user", "content": prompt}
    ]
    # Call the model in a minimal, non-streaming fashion; hard-fallback to 'other' on any error.
    content = None
    try:
        result = model.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=8,
            top_p=1.0,
            stream=False
        )

        if isinstance(result, dict) and "choices" in result:
            first_choice = result["choices"][0]
            if isinstance(first_choice, dict) and "message" in first_choice and "content" in first_choice["message"]:
                content = first_choice["message"]["content"]
            elif isinstance(first_choice, dict) and "text" in first_choice:
                content = first_choice["text"]
            else:
                content = str(first_choice)
        else:
            tokens = []
            for chunk in result:
                if isinstance(chunk, dict):
                    if "delta" in chunk and isinstance(chunk["delta"], dict) and "content" in chunk["delta"]:
                        tokens.append(chunk["delta"]["content"])
                    elif "choices" in chunk:
                        c = chunk["choices"][0]
                        if isinstance(c, dict):
                            tokens.append(c.get("message", {}).get("content", "") or c.get("text", ""))
                        else:
                            tokens.append(str(c))
                    else:
                        tokens.append(str(chunk))
                else:
                    tokens.append(str(chunk))
            content = "".join(tokens)
    except AssertionError:
        logger.exception("LLM backend assertion during intent detection; falling back to 'other'.")
        return "other"
    except Exception:
        logger.exception("LLM call/iteration failed during intent detection; falling back to 'other'.")
        return "other"
    # Accept only the first word or intent string; guard against whitespace-only content
    tokens = (content or "").strip().split()
    intent = tokens[0] if tokens else "other"
    intent = intent if intent in {
        "read_file", "write_file", "mark_task_complete", "reflect", "seed", "shell", "list_tools", "list_tree", "find_file", "list_files", "recitation", "tool_routing", "clarification", "correction", "brainstorming", "feedback", "chat", "other"
    } else "other"

    # Runtime heuristic: override spurious read/write classifications unless
    # the user actually referenced files/logs.  This keeps ordinary questions
    # (e.g., "Who forged Excalibur?") on the conversation track instead of
    # forcing the operator to touch intent overrides.
    if intent in {"read_file", "write_file"} and not _mentions_file_like_action(text):
        logger.info("Intent heuristic override: '%s' lacks file keywords; downgrading to 'other'.", intent)
        intent = "other"

    if intent == "other" and _detect_direct_list_tools(text):
        intent = "list_tools"

    logger.info(f"Model intent detection: {intent}")
    return intent

# ---- Unified entrypoint ----
def detect_intent(text, config, lite_llm=None, full_llm=None, fallback_llm=None, probe_context="") -> Plan:
    """
    Detects intent using reflex path, else LLM.
    Returns a Plan object.

    Args:
        probe_context: Optional domain hint from semantic probe for LLM-based classification.
    """
    try:
        logger.info("Intent detection: start")
        logger.debug("[DEBUG] Intent detect start len=%d", len(text or ""))
    except Exception:
        logger.debug("[DEBUG] Intent detect start")
    # 1. Reflex check
    intent_str = fast_intent_check(text)
    if not intent_str:
        # 2. LLM path
        intent_str = model_intent_detection(text, config, lite_llm, full_llm, fallback_llm, probe_context=probe_context)
    
    read_only_intents = {"read_file", "explain_file", "explain_symbol"}
    plan = Plan(intent=intent_str, read_only=intent_str in read_only_intents)
    
    logger.debug(f"Detected plan: {plan}")
    logger.info("Intent detection: done intent=%s", plan.intent)
    print(f"INTENT_DETECTED: {plan.intent}", file=sys.stderr)
    logger.debug("[DEBUG] Intent detect done intent=%s read_only=%s", plan.intent, plan.read_only)
    return plan
