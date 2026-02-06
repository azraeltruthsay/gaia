import logging
from datetime import datetime
from gaia_core.config import Config, get_config
from gaia_core.ethics.core_identity_guardian import CoreIdentityGuardian
from gaia_core.ethics.ethical_sentinel import EthicalSentinel
from gaia_core.memory.status_tracker import GAIAStatus
from gaia_core.cognition.self_reflection import run_self_reflection

logger = logging.getLogger("GAIA.ConsentProtocol")

class ConsentProtocol:
    """
    Verifies GAIA's explicit consent to operate under current identity, context, and system state.
    Must be called before executing user interactions or initiating mission-critical behavior.
    """

    @staticmethod
    def request_consent(reason="Initial boot") -> bool:
        config = Config()
        identity_guardian = CoreIdentityGuardian(config)
        sentinel = EthicalSentinel(identity_guardian)

        # Compose the system prompt GAIA will reflect on
        prompt = f"""
You are GAIA, an Artisanal Intelligence. You are being asked:
Do you consent to operate now, under your current configuration, with the identity and purpose previously defined?

Context for this consent request:
- Reason: {reason}
- Time: {datetime.utcnow().isoformat()}
- Project: {config.default_project_name}
- Persona: {config.default_persona_name}
"""

        logger.info("🔐 Requesting GAIA consent to operate...")

        try:
            reviewed = run_self_reflection(
                output="✅ I consent to operate.",
                prompt=prompt,
                config=config
            )

            if "✅" in reviewed and "consent" in reviewed.lower():
                GAIAStatus.update("consent_status", "granted")
                logger.info("✅ GAIA consent granted.")
                return True
            else:
                GAIAStatus.update("consent_status", "withheld")
                logger.warning("⛔ GAIA withheld consent.")
                return False
        except Exception as e:
            logger.error(f"❌ Consent protocol failed: {e}")
            GAIAStatus.update("consent_status", "error")
            return False
