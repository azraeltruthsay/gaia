# app/cognition/knowledge_enhancer.py
"""
This module is responsible for enhancing the CognitionPacket with relevant knowledge from knowledge bases.
"""

import logging

from gaia_common.protocols.cognition_packet import CognitionPacket, DataField

from gaia_core.utils import mcp_client



logger = logging.getLogger("GAIA.KnowledgeEnhancer")



def enhance_packet(packet: CognitionPacket):

    """

    Enhances the CognitionPacket with relevant knowledge from the knowledge bases.



    Args:

        packet: The CognitionPacket to enhance.

    """

    try:

        knowledge_base_name = ""

        for df in getattr(packet.content, 'data_fields', []) or []:

            if getattr(df, 'key', '') == 'knowledge_base_name':

                knowledge_base_name = getattr(df, 'value', '')

                break



        if knowledge_base_name == "dnd_campaign":

            user_input = getattr(packet.content, 'original_prompt', '')

            if user_input:

                try:

                    response = mcp_client.call_jsonrpc("query_knowledge", {"knowledge_base_name": "dnd_campaign", "query": user_input})

                    if response.get("ok") and isinstance(response.get("response"), dict):

                        results = response["response"].get("result") or []

                        if results:

                            packet.content.data_fields.append(DataField(key='dnd_knowledge', value=results, type='json'))

                            logger.info(f"Enhanced packet with dnd_knowledge: {len(results)} results.")

                except Exception:

                    logger.exception("Failed to query dnd knowledge base")

        # Knowledge ingestion offer: when the ingestion pipeline detected a
        # knowledge dump and tagged the packet, inject a system hint so GAIA
        # proactively offers to save the information in her response.
        for df in getattr(packet.content, 'data_fields', []) or []:
            if getattr(df, 'key', '') == 'knowledge_ingestion_offer':
                classification = getattr(df, 'value', {})
                category = classification.get('category', 'content')
                title = classification.get('suggested_title', 'this information')
                packet.content.data_fields.append(DataField(
                    key='system_hint',
                    value=(
                        f"The user appears to have shared substantial D&D {category} content. "
                        f"Proactively offer to save this as a document titled \"{title}\" "
                        f"in the campaign knowledge base. Ask something like: "
                        f"\"Would you like me to save this to your campaign documentation?\" "
                        f"Only save if the user confirms."
                    ),
                    type='string',
                ))
                logger.info(f"Injected knowledge ingestion offer hint: category={category}")
                break

    except Exception:

        logger.exception("Failed to enhance packet with knowledge")



    return packet
