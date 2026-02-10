"""
DevMatrixAnalyzer - Analyzes task completion status for GAIA's dev_matrix.

This module provides automated task completion detection for GAIA's self-development
roadmap. Each task type has specific verification logic.
"""

import logging
from pathlib import Path
from typing import Dict, List, Callable
from gaia_core.memory.dev_matrix import GAIADevMatrix

logger = logging.getLogger("GAIA.DevMatrixAnalyzer")


class DevMatrixAnalyzer:
    """
    Analyzes and updates dev_matrix task completion status.

    Uses file-based verification instead of shell commands for safety and portability.
    """

    def __init__(self, config):
        self.config = config
        self.dev_matrix = GAIADevMatrix(config)
        self._base_path = Path(getattr(config, 'base_path', '/gaia/GAIA_Project/gaia-assistant'))

        # Registry of task verification functions
        # Maps task name (or partial match) to a verification function
        self._task_verifiers: Dict[str, Callable[[], bool]] = {
            "Discord integration": self._verify_discord_integration,
            "Thought seed creation tooling": self._verify_thought_seed_tooling,
            "Self-reflection triggers real code review": self._verify_self_reflection,
            "GCP packet fragmentation": self._verify_gcp_fragmentation,
        }

    def analyze_and_update(self) -> List[Dict]:
        """
        Analyze all open tasks and update their status if completed.

        Returns:
            List of tasks that were marked as resolved in this run.
        """
        newly_resolved = []
        open_tasks = self.dev_matrix.get_open_tasks()

        for task in open_tasks:
            task_name = task.get('task', '')
            if self.is_task_completed(task):
                logger.info(f"DevMatrixAnalyzer: Task '{task_name}' detected as completed")
                if self.dev_matrix.resolve_task(task_name):
                    newly_resolved.append(task)
                    logger.info(f"DevMatrixAnalyzer: Marked '{task_name}' as resolved")

        return newly_resolved

    def is_task_completed(self, task: Dict) -> bool:
        """
        Check if a specific task is completed.

        Args:
            task: Task dict from dev_matrix with 'task' key

        Returns:
            True if task appears to be completed based on codebase analysis
        """
        task_name = task.get('task', '')

        # Check direct match first
        if task_name in self._task_verifiers:
            try:
                return self._task_verifiers[task_name]()
            except Exception as e:
                logger.warning(f"DevMatrixAnalyzer: Verification failed for '{task_name}': {e}")
                return False

        # Check partial matches
        for pattern, verifier in self._task_verifiers.items():
            if pattern.lower() in task_name.lower():
                try:
                    return verifier()
                except Exception as e:
                    logger.warning(f"DevMatrixAnalyzer: Verification failed for '{task_name}' (pattern: {pattern}): {e}")
                    return False

        # No verifier found - can't auto-complete
        return False

    def get_task_status_report(self) -> Dict:
        """
        Generate a status report for all tasks.

        Returns:
            Dict with 'open', 'resolved', and 'verifiable' task lists
        """
        all_tasks = self.dev_matrix.dump()
        open_tasks = [t for t in all_tasks if t.get('status') == 'open']
        resolved_tasks = [t for t in all_tasks if t.get('status') == 'resolved']

        # Check which open tasks have verifiers
        verifiable = []
        for task in open_tasks:
            task_name = task.get('task', '')
            has_verifier = task_name in self._task_verifiers or any(
                p.lower() in task_name.lower() for p in self._task_verifiers
            )
            if has_verifier:
                verifiable.append(task_name)

        return {
            'total': len(all_tasks),
            'open': len(open_tasks),
            'resolved': len(resolved_tasks),
            'open_tasks': [t.get('task') for t in open_tasks],
            'verifiable_tasks': verifiable,
            'recently_resolved': [t.get('task') for t in resolved_tasks[-5:]],
        }

    # --- Task-specific verification functions ---

    def _verify_discord_integration(self) -> bool:
        """
        Verify Discord integration is complete.

        Checks:
        1. DiscordConnector class exists
        2. Bot listener functionality is implemented
        3. DM support is implemented
        4. Integration with AgentCore exists
        """
        connector_path = self._base_path / "app" / "integrations" / "discord_connector.py"

        if not connector_path.exists():
            logger.debug("Discord connector file not found")
            return False

        try:
            content = connector_path.read_text()

            # Check for key implementation markers
            required_markers = [
                "class DiscordConnector",        # Main class
                "def start_bot_listener",        # Bot listener
                "is_dm",                         # DM support
                "def send(",                     # Send capability
                "_message_callback",             # Message routing to AgentCore
            ]

            missing = [m for m in required_markers if m not in content]
            if missing:
                logger.debug(f"Discord integration missing markers: {missing}")
                return False

            # Check that rescue shell has Discord integration
            rescue_path = self._base_path / "gaia_rescue.py"
            if rescue_path.exists():
                rescue_content = rescue_path.read_text()
                if "discord" not in rescue_content.lower():
                    logger.debug("Discord not integrated into rescue shell")
                    return False

            logger.info("Discord integration verification passed")
            return True

        except Exception as e:
            logger.warning(f"Discord verification error: {e}")
            return False

    def _verify_thought_seed_tooling(self) -> bool:
        """Verify thought seed creation tooling exists."""
        seed_path = self._base_path / "app" / "cognition" / "thought_seed.py"

        if not seed_path.exists():
            return False

        try:
            content = seed_path.read_text()
            return "save_thought_seed" in content and "class" in content.lower()
        except Exception:
            return False

    def _verify_self_reflection(self) -> bool:
        """Verify self-reflection triggers code review."""
        reflection_path = self._base_path / "app" / "cognition" / "self_reflection.py"

        if not reflection_path.exists():
            return False

        try:
            content = reflection_path.read_text()
            return "run_self_reflection" in content or "reflect_and_refine" in content
        except Exception:
            return False

    def _verify_gcp_fragmentation(self) -> bool:
        """Verify GCP packet fragmentation is implemented."""
        agent_core_path = self._base_path / "app" / "cognition" / "agent_core.py"

        if not agent_core_path.exists():
            return False

        try:
            content = agent_core_path.read_text()
            # Check for fragmentation-related code
            markers = ["fragment", "assemble", "sketchpad"]
            return sum(1 for m in markers if m in content.lower()) >= 2
        except Exception:
            return False


def analyze_dev_matrix(config) -> Dict:
    """
    Convenience function to run a full dev_matrix analysis.

    Args:
        config: GAIA config object

    Returns:
        Status report dict
    """
    analyzer = DevMatrixAnalyzer(config)
    newly_resolved = analyzer.analyze_and_update()
    report = analyzer.get_task_status_report()
    report['newly_resolved'] = [t.get('task') for t in newly_resolved]
    return report
