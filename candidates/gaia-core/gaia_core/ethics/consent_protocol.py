import logging
from datetime import datetime, timezone
from gaia_core.config import Config
from gaia_core.ethics.core_identity_guardian import CoreIdentityGuardian
from gaia_core.ethics.ethical_sentinel import EthicalSentinel
from gaia_core.memory.status_tracker import GAIAStatus
from gaia_core.cognition.self_reflection import run_self_reflection
from gaia_core.cognition.packet_utils import upgrade_v2_to_v3_packet

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
        EthicalSentinel(identity_guardian)

        # Compose the system prompt GAIA will reflect on
        prompt = f"""
You are GAIA, an Artisanal Intelligence. You are being asked:
Do you consent to operate now, under your current configuration, with the identity and purpose previously defined?

Context for this consent request:
- Reason: {reason}
- Time: {datetime.now(timezone.utc).isoformat()}
- Project: {config.default_project_name}
- Persona: {config.default_persona_name}
"""

        logger.info("🔐 Requesting GAIA consent to operate...")

        try:
            # o9dh/saq3: run_self_reflection requires a CognitionPacket
            # (packet.content.original_prompt becomes what's reflected on)
            # and has no `prompt` kwarg at all -- this call was raising
            # TypeError on every invocation, caught below, so GAIA's boot
            # consent check has always silently gone to the except branch
            # ("error" status, consent withheld) rather than actually
            # running. Fold the carefully-built context into `output`
            # (what gets reflected on) since there's no separate slot for
            # it, and build a minimal valid packet via the same converter
            # used elsewhere for legacy/synthetic packets.
            reflection_packet = upgrade_v2_to_v3_packet({})
            reviewed = run_self_reflection(
                packet=reflection_packet,
                output=f"{prompt}\n\n✅ I consent to operate.",
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
