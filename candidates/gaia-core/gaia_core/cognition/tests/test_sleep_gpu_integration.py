"""
Tests for sleep cycle GPU release/reclaim and Discord presence wiring.

Validates that SleepCycleLoop correctly:
- Calls orchestrator /gpu/sleep when entering sleep
- Calls orchestrator /gpu/wake when waking
- Continues gracefully when orchestrator is unreachable
- Sets Discord to yellow dot (idle) during sleep
- Resets Discord to green (online) on wake
- Sets Discord to dnd during DREAMING / DISTRACTED
"""

from unittest.mock import MagicMock, patch
import pytest

from gaia_core.cognition.sleep_cycle_loop import SleepCycleLoop
from gaia_core.cognition.sleep_wake_manager import GaiaState, _TransientPhase


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.SLEEP_ENABLED = True
    config.SLEEP_IDLE_THRESHOLD_MINUTES = 5
    config.SLEEP_CHECKPOINT_DIR = "/tmp/test_sleep"
    config.SLEEP_ENABLE_QLORA = False
    config.SLEEP_ENABLE_DREAM = False
    config.SLEEP_TASK_TIMEOUT = 600
    return config


@pytest.fixture
def mock_discord():
    connector = MagicMock()
    connector.update_presence = MagicMock()
    connector.set_idle = MagicMock()
    return connector


@pytest.fixture
def loop(mock_config, mock_discord):
    """Create a SleepCycleLoop with mocked dependencies."""
    with patch("gaia_core.cognition.sleep_cycle_loop.SleepWakeManager"), \
         patch("gaia_core.cognition.sleep_task_scheduler.SleepTaskScheduler"):
        scl = SleepCycleLoop(
            mock_config,
            discord_connector=mock_discord,
            model_pool=None,
            agent_core=None,
        )
        scl._orchestrator_url = "http://test-orchestrator:6410"
        scl._web_url = "http://test-web:6414"
    return scl


# =============================================================================
# GPU release on sleep
# =============================================================================

class TestGPUReleaseOnSleep:
    def test_gpu_release_called_on_sleep(self, loop):
        """After successful initiate_drowsy(), orchestrator /gpu/sleep is called."""
        loop.sleep_wake_manager.should_transition_to_drowsy.return_value = True
        loop.sleep_wake_manager.initiate_drowsy.return_value = True

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.post.return_value = mock_resp

            loop._handle_active(6.0)

            mock_httpx.post.assert_any_call(
                "http://test-orchestrator:6410/gpu/sleep",
                json={"reason": "sleep_cycle"},
                timeout=60.0,
            )

    def test_gpu_release_failure_nonfatal(self, loop):
        """If orchestrator is unreachable, sleep still proceeds."""
        loop.sleep_wake_manager.should_transition_to_drowsy.return_value = True
        loop.sleep_wake_manager.initiate_drowsy.return_value = True

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            mock_httpx.post.side_effect = ConnectionError("unreachable")

            # Should not raise
            loop._handle_active(6.0)

            # State machine still transitioned (initiate_drowsy returned True)
            loop.sleep_wake_manager.initiate_drowsy.assert_called_once()

    def test_no_gpu_release_when_drowsy_cancelled(self, loop):
        """If drowsy is cancelled (wake arrived), GPU release is NOT called."""
        loop.sleep_wake_manager.should_transition_to_drowsy.return_value = True
        loop.sleep_wake_manager.initiate_drowsy.return_value = False

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            loop._handle_active(6.0)

            # /gpu/sleep should never be called since drowsy was cancelled
            for call in mock_httpx.post.call_args_list:
                assert "/gpu/sleep" not in str(call)


# =============================================================================
# GPU reclaim on wake (via _handle_asleep with WAKING phase)
# =============================================================================

class TestGPUReclaimOnWake:
    def test_gpu_reclaim_called_on_wake(self, loop):
        """During waking phase, orchestrator /gpu/wake is called."""
        loop.sleep_wake_manager._phase = _TransientPhase.WAKING
        loop.sleep_wake_manager.complete_wake.return_value = {"checkpoint_loaded": True}

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.post.return_value = mock_resp

            loop._handle_asleep()

            mock_httpx.post.assert_any_call(
                "http://test-orchestrator:6410/gpu/wake",
                json={},
                timeout=180.0,
            )

    def test_gpu_reclaim_failure_nonfatal(self, loop):
        """If GPU reclaim fails, complete_wake() still runs."""
        loop.sleep_wake_manager._phase = _TransientPhase.WAKING
        loop.sleep_wake_manager.complete_wake.return_value = {}

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            mock_httpx.post.side_effect = ConnectionError("unreachable")

            loop._handle_asleep()

            loop.sleep_wake_manager.complete_wake.assert_called_once()


# =============================================================================
# Discord presence during sleep
# =============================================================================

class TestPresenceDuringSleep:
    def test_presence_sleeping_during_sleep(self, loop, mock_discord):
        """Discord shows yellow dot + 'sleeping...' when entering sleep."""
        loop.sleep_wake_manager.should_transition_to_drowsy.return_value = True
        loop.sleep_wake_manager.initiate_drowsy.return_value = True

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.post.return_value = mock_resp

            loop._handle_active(6.0)

        mock_discord.update_presence.assert_called_with("sleeping...", status_override="idle")

    def test_presence_sleeping_during_task(self, loop, mock_discord):
        """Discord shows yellow dot + task type during sleep tasks."""
        loop.sleep_wake_manager._phase = _TransientPhase.NONE
        loop._resource_monitor = None  # disable distracted check
        task = MagicMock()
        task.task_id = "test_task"
        task.task_type = "conversation_curation"
        task.interruptible = True
        loop.sleep_task_scheduler.get_next_task.return_value = task
        loop.sleep_wake_manager.wake_signal_pending = False

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx"):
            loop._handle_asleep()

        mock_discord.update_presence.assert_called_with(
            "sleeping: conversation_curation", status_override="idle"
        )

    def test_presence_resets_on_wake(self, loop, mock_discord):
        """Discord returns to green (set_idle) after waking."""
        loop.sleep_wake_manager._phase = _TransientPhase.WAKING
        loop.sleep_wake_manager.complete_wake.return_value = {}

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_httpx.post.return_value = mock_resp

            loop._handle_asleep()

        mock_discord.set_idle.assert_called_once()


# =============================================================================
# DREAMING presence
# =============================================================================

class TestDreamingPresence:
    def test_dreaming_shows_dnd_studying(self, loop, mock_discord):
        """During DREAMING, Discord shows dnd + 'studying...'."""
        loop._handle_dreaming()
        mock_discord.update_presence.assert_called_with("studying...", status_override="dnd")


# =============================================================================
# SOA mode presence (no discord_connector, uses gaia-web HTTP)
# =============================================================================

class TestSOAPresence:
    def test_soa_presence_calls_web_endpoint(self, mock_config):
        """Without discord_connector, presence updates go to gaia-web /presence."""
        with patch("gaia_core.cognition.sleep_cycle_loop.SleepWakeManager"), \
             patch("gaia_core.cognition.sleep_task_scheduler.SleepTaskScheduler"):
            scl = SleepCycleLoop(mock_config, discord_connector=None)
            scl._web_url = "http://test-web:6414"

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            scl._update_presence("sleeping...", sleeping=True)

            mock_httpx.post.assert_called_with(
                "http://test-web:6414/presence",
                json={"activity": "sleeping...", "status": "idle"},
                timeout=5.0,
            )

    def test_soa_presence_online_when_not_sleeping(self, mock_config):
        """Non-sleeping presence updates don't include status=idle."""
        with patch("gaia_core.cognition.sleep_cycle_loop.SleepWakeManager"), \
             patch("gaia_core.cognition.sleep_task_scheduler.SleepTaskScheduler"):
            scl = SleepCycleLoop(mock_config, discord_connector=None)
            scl._web_url = "http://test-web:6414"

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            scl._update_presence("Waking up...")

            mock_httpx.post.assert_called_with(
                "http://test-web:6414/presence",
                json={"activity": "Waking up..."},
                timeout=5.0,
            )

    def test_soa_presence_failure_nonfatal(self, mock_config):
        """If gaia-web is unreachable, presence update fails silently."""
        with patch("gaia_core.cognition.sleep_cycle_loop.SleepWakeManager"), \
             patch("gaia_core.cognition.sleep_task_scheduler.SleepTaskScheduler"):
            scl = SleepCycleLoop(mock_config, discord_connector=None)
            scl._web_url = "http://test-web:6414"

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            mock_httpx.post.side_effect = ConnectionError("unreachable")
            # Should not raise
            scl._update_presence("sleeping...", sleeping=True)

    def test_soa_presence_invisible_on_offline(self, mock_config):
        """Offline presence sends invisible status to gaia-web."""
        with patch("gaia_core.cognition.sleep_cycle_loop.SleepWakeManager"), \
             patch("gaia_core.cognition.sleep_task_scheduler.SleepTaskScheduler"):
            scl = SleepCycleLoop(mock_config, discord_connector=None)
            scl._web_url = "http://test-web:6414"

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            scl._update_presence(None, offline=True)

            mock_httpx.post.assert_called_with(
                "http://test-web:6414/presence",
                json={"activity": "over the studio", "status": "invisible"},
                timeout=5.0,
            )

    def test_soa_presence_dnd_override(self, mock_config):
        """Status override sends explicit status to gaia-web."""
        with patch("gaia_core.cognition.sleep_cycle_loop.SleepWakeManager"), \
             patch("gaia_core.cognition.sleep_task_scheduler.SleepTaskScheduler"):
            scl = SleepCycleLoop(mock_config, discord_connector=None)
            scl._web_url = "http://test-web:6414"

        with patch("gaia_core.cognition.sleep_cycle_loop.httpx") as mock_httpx:
            scl._update_presence("studying...", status_override="dnd")

            mock_httpx.post.assert_called_with(
                "http://test-web:6414/presence",
                json={"activity": "studying...", "status": "dnd"},
                timeout=5.0,
            )
