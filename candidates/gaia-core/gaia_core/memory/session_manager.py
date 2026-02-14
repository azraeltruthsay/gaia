import os
import glob
import json
import logging
import threading
from typing import List, Dict
from datetime import datetime, timedelta
from gaia_core.config import Config, get_config

# Import the specialist tools the manager will orchestrate
from gaia_core.memory.conversation.summarizer import ConversationSummarizer
from gaia_core.memory.conversation.keywords import ConversationKeywordExtractor
from gaia_core.memory.conversation.archiver import ConversationArchiver
from gaia_core.utils.output_router import _strip_think_tags_robust

logger = logging.getLogger("GAIA.SessionManager")

# A single, central file to store the state of all sessions.
# This allows different processes (web, cli) to share conversation state.
STATE_FILE = "app/shared/sessions.json"
_lock = threading.Lock()  # Prevents file corruption from simultaneous writes
# MODIFICATION: Add a constant for the new timestamp file
LAST_ACTIVITY_FILE = "app/shared/last_activity.timestamp"

class Session:
    """
    A dedicated data class to hold the state for a single conversation session.
    Using a class instead of a raw dictionary improves code clarity and robustness.
    """

    def __init__(self, session_id: str, persona: str = "default"):
        self.session_id: str = session_id
        self.persona: str = persona
        self.history: List[Dict] = []
        self.created_at: datetime = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Serializes the session object to a dictionary for JSON storage."""
        return {
            "session_id": self.session_id,
            "persona": self.persona,
            "history": self.history,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Session':
        """Deserializes a dictionary from the state file back into a Session object."""
        session = cls(session_id=data["session_id"], persona=data.get("persona", "default"))
        session.history = data.get("history", [])
        try:
            # Handle ISO format strings for datetime objects
            session.created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        except (TypeError, ValueError):
            logger.warning(f"Could not parse 'created_at' for session {session.session_id}, using current time.")
            session.created_at = datetime.utcnow()
        return session


class SessionManager:
    """
    Manages loading, saving, and accessing all persistent conversation sessions.
    This is the single source of truth for conversation state, designed to be
    process-safe and to orchestrate long-term memory functions.
    """

    def __init__(self, config, llm=None, embed_model=None):
        self.config = config
        self.sessions: Dict[str, Session] = self._load_state()

        # Initialize the specialist tools the manager will use
        self.summarizer = ConversationSummarizer(llm=llm, embed_model=embed_model)
        self.keyword_extractor = ConversationKeywordExtractor()
        self.archiver = ConversationArchiver(config)

        # Define the threshold for when to trigger long-term memory processing
        self.max_active_messages = 20
        logger.info(f"SessionManager initialized. Found {len(self.sessions)} existing sessions.")

    def _load_state(self) -> Dict[str, Session]:
        """Loads all sessions from the central state file in a thread-safe manner."""
        with _lock:
            try:
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        logger.info(f"💾 Loading {len(data)} sessions from {STATE_FILE}")
                        return {sid: Session.from_dict(sdata) for sid, sdata in data.items()}
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"❌ Could not load state file {STATE_FILE}. Starting fresh. Error: {e}")
        return {}

    @staticmethod
    def _sanitize_for_json(obj):
        """Recursively convert non-serializable objects to strings."""
        if isinstance(obj, dict):
            return {k: SessionManager._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [SessionManager._sanitize_for_json(v) for v in obj]
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # Generators, objects, etc. — force to string
        return str(obj)

    def _save_state(self):
        """Saves the current state of all sessions to the file in a thread-safe manner."""
        with _lock:
            try:
                # Ensure the directory exists before writing
                os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
                with open(STATE_FILE, 'w', encoding='utf-8') as f:
                    # Serialize all session objects into a dictionary before saving
                    data_to_save = {sid: session.to_dict() for sid, session in self.sessions.items()}
                    data_to_save = self._sanitize_for_json(data_to_save)
                    json.dump(data_to_save, f, indent=2)
                    logger.debug(f"💾 Session state saved to {STATE_FILE}")
            except IOError as e:
                logger.error(f"❌ Could not save state file {STATE_FILE}. Error: {e}")

    def get_or_create_session(self, session_id: str, persona: str = "default") -> Session:
        """Retrieves a session by ID or creates a new one if it doesn't exist."""
        if session_id not in self.sessions:
            logger.info(f"✨ Creating new persistent session: {session_id} with persona '{persona}'")
            self.sessions[session_id] = Session(session_id, persona)
            self._save_state()  # Save immediately after creation
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """Adds a message and checks if it's time to create a long-term memory."""
        # Defense-in-depth: strip think/reasoning tags from assistant messages
        # before persisting, so poisoned history can't confuse future model calls.
        if role == "assistant":
            content = _strip_think_tags_robust(content)
            if not content.strip():
                logger.warning(f"Skipping empty assistant message for session '{session_id}' (was only think tags)")
                return
        session = self.get_or_create_session(session_id)
        session.history.append({"role": role, "content": content})
        logger.debug(f"💬 Added '{role}' message to session '{session_id}'. History length: {len(session.history)}")

        # Index completed turn-pairs for session RAG retrieval
        if role == "assistant" and len(session.history) >= 2:
            try:
                from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
                indexer = SessionHistoryIndexer.instance(session_id)
                user_msg = session.history[-2].get("content", "")
                assistant_msg = session.history[-1].get("content", "")
                turn_idx = len(session.history) // 2 - 1
                indexer.index_turn(turn_idx, user_msg, assistant_msg)
            except Exception as e:
                logger.warning(f"Session indexing failed (non-fatal): {e}")

        # Check if the conversation is long enough to be archived
        if len(session.history) >= self.max_active_messages:
            self.summarize_and_archive(session_id)
        else:
            # If not archiving, just save the new message to the state file
            self._save_state()

    def summarize_and_archive(self, session_id: str):
        """
        Orchestrates the specialists to process a session's history,
        create a long-term memory, and then clear the active history.
        """
        session = self.get_or_create_session(session_id)
        if not session.history:
            logger.warning(f"Attempted to archive session '{session_id}', but history is empty.")
            return

        try:
            logger.info(f"🗃️ Session '{session_id}' reached message limit. Creating long-term memory.")

            # 1. Delegate to the Summarizer; build a compact packet snapshot to ground the summary
            try:
                from gaia_core.utils.packet_builder import build_packet_snapshot
                packet_snapshot = build_packet_snapshot(session_id=session.session_id, persona_id=session.persona, original_prompt=session.history[-1]['content'] if session.history else '')
            except Exception:
                packet_snapshot = None
            summary = self.summarizer.generate_summary(session.history, packet=packet_snapshot)

            # 2. Delegate to the Keyword Extractor
            keywords = self.keyword_extractor.extract_keywords(session.history)

            # 3. Delegate to the Archiver
            self.archiver.archive_conversation(
                session_id=session.session_id,
                persona=session.persona,
                messages=session.history,
                summary=summary,
                keywords=keywords
            )

            # 3.5 Archive session vector index alongside conversation archive
            try:
                from gaia_core.memory.session_history_indexer import SessionHistoryIndexer
                indexer = SessionHistoryIndexer.instance(session_id)
                indexer.archive_and_reset()
            except Exception:
                logger.debug("Session vector index archive failed (non-fatal)", exc_info=True)

            # 3.6 Curate notable conversations for knowledge examples
            try:
                from gaia_core.cognition.conversation_curator import ConversationCurator
                curator = ConversationCurator()
                curator.curate(session.session_id, session.history)
            except Exception:
                logger.debug("Conversation curation failed (non-fatal)", exc_info=True)

            # 4. Clear the active history to manage context size for the next turn
            session.history.clear()
            logger.info(f"✅ Active history for session '{session_id}' cleared after successful archiving.")

        except Exception as e:
            logger.error(f"❌ Failed to archive session '{session_id}': {e}", exc_info=True)
        finally:
            # Always save the state. This persists the cleared history.
            self._save_state()

    def get_history(self, session_id: str) -> List[Dict]:
        """Returns the full message history for a given session."""
        session = self.get_or_create_session(session_id)
        return session.history

    def reset_session(self, session_id: str):
        """Deletes a session from the state, effectively resetting it."""
        if session_id in self.sessions:
            logger.info(f"🔄 Resetting session: {session_id}")
            del self.sessions[session_id]
            self._save_state()
        else:
            logger.warning(f"Attempted to reset non-existent session: {session_id}")

    def sanitize_sessions(
        self,
        vector_dir: str = "data/shared/session_vectors",
        max_age_days: int = 7,
        max_active_messages: int = 0,
    ) -> Dict[str, int]:
        """
        Clean up stale, orphaned, and test session artifacts.

        Called on startup (e.g. Discord on_ready) to keep session state lean.

        Steps:
          1. Remove smoke-test-* and test-* sessions from in-memory state
          2. Delete orphaned vector files (no matching session in state)
          3. Delete smoke-test-* and test-* vector files unconditionally
          4. Optionally trim sessions older than max_age_days with empty history
          5. Persist cleaned state

        Args:
            vector_dir: Path to session vector index files
            max_age_days: Sessions older than this with empty history are removed
            max_active_messages: If >0, sessions exceeding this trigger archival

        Returns:
            Dict with counts: sessions_purged, vectors_purged, smoke_purged
        """
        counts = {"sessions_purged": 0, "vectors_purged": 0, "smoke_purged": 0}
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)

        # ── Step 1: Purge test/smoke sessions from in-memory state ──
        to_remove = []
        for sid in list(self.sessions.keys()):
            if sid.startswith("smoke-test-") or sid.startswith("test-"):
                to_remove.append(sid)
            elif self.sessions[sid].created_at < cutoff and not self.sessions[sid].history:
                # Old session with empty history — stale shell
                to_remove.append(sid)

        for sid in to_remove:
            del self.sessions[sid]
            counts["sessions_purged"] += 1
            logger.info(f"Sanitize: removed session '{sid}'")

        # ── Step 2: Clean vector files ──
        if os.path.isdir(vector_dir):
            active_sids = set(self.sessions.keys())
            for vec_file in glob.glob(os.path.join(vector_dir, "*.json")):
                basename = os.path.basename(vec_file)
                vec_sid = basename.replace(".json", "")

                is_smoke = vec_sid.startswith("smoke-test-")
                is_test = vec_sid.startswith("test-")
                is_orphan = vec_sid not in active_sids

                if is_smoke or is_test:
                    try:
                        os.remove(vec_file)
                        counts["smoke_purged"] += 1
                        logger.info(f"Sanitize: deleted smoke/test vector {basename}")
                    except OSError as e:
                        logger.warning(f"Sanitize: could not delete {basename}: {e}")
                elif is_orphan:
                    try:
                        os.remove(vec_file)
                        counts["vectors_purged"] += 1
                        logger.info(f"Sanitize: deleted orphaned vector {basename}")
                    except OSError as e:
                        logger.warning(f"Sanitize: could not delete {basename}: {e}")

        # ── Step 3: Persist cleaned state ──
        if counts["sessions_purged"] > 0:
            self._save_state()

        logger.info(
            f"Session sanitization complete: "
            f"{counts['sessions_purged']} sessions purged, "
            f"{counts['vectors_purged']} orphaned vectors removed, "
            f"{counts['smoke_purged']} smoke/test vectors removed"
        )
        return counts

    # MODIFICATION: Add a new method to record the last system activity
    def record_last_activity(self):
        """
        Updates a timestamp file with the current time.
        This serves as a signal to the GIL that the system is not idle.
        """
        with _lock:  # Use the same lock to prevent race conditions
            try:
                os.makedirs(os.path.dirname(LAST_ACTIVITY_FILE), exist_ok=True)
                with open(LAST_ACTIVITY_FILE, 'w', encoding='utf-8') as f:
                    f.write(datetime.utcnow().isoformat())
                logger.debug(f"Timestamp updated in {LAST_ACTIVITY_FILE}")
            except IOError as e:
                logger.error(f"❌ Could not write to last activity file {LAST_ACTIVITY_FILE}: {e}")