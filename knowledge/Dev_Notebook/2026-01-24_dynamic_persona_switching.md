# Dev Journal - January 24, 2026

## Objective

Implement a dynamic persona switching mechanism within GAIA. The goal is to allow GAIA to default to a generalist "dev" persona but intelligently switch to a specialized persona based on the user's detected intent, and then switch back when the context changes.

The initial use case will be for Dungeons & Dragons conversations. When a user asks about D&D, GAIA will adopt a "D&D player assistant" persona.

## Plan

1.  **Create a "D&D Player Assistant" Persona:**
    *   A new persona file will be created under `knowledge/personas/` named `dnd_player_assistant.json`.
    *   This persona will be designed to be helpful and knowledgeable about D&D from a player's perspective, distinct from the DM persona (`strauthauk`).

2.  **Generalize the Discord Message Handler:**
    *   The `handle_discord_message` function in `gaia_rescue.py` will be simplified.
    *   The hardcoded logic that checks for the `dnd-campaign` project and immediately queries the knowledge base will be removed.
    *   Instead, all messages will be passed to the `AgentCore` for processing, ensuring a consistent entry point.

3.  **Implement Intent-Based Persona Switching in `AgentCore`:**
    *   A new mechanism will be added to `app/cognition/agent_core.py` to manage persona switching.
    *   A simple, keyword-based intent detection will be implemented directly within `AgentCore` for the initial version. It will scan for D&D-related keywords (e.g., "d&d", "dnd", "character sheet", "spell").
    *   When D&D intent is detected, the `AgentCore` will dynamically load the `dnd_player_assistant` persona and apply it to the current `CognitionPacket`.
    *   The system will default to the "dev" persona. If a D&D-related message is not detected, it will ensure the "dev" persona is active.

4.  **Update `CognitionPacket` and `PromptBuilder`:**
    *   The `CognitionPacket` will be the primary carrier of the active persona state for a given turn.
    *   The `PromptBuilder` in `app/utils/prompt_builder.py` will be updated to use the persona information from the packet to construct the final system prompt. This ensures the model receives the correct instructions for its current persona.

## Implementation Details

*   **Persona Switching Logic:** The `AgentCore.run_turn` method will be the central point for this logic. Before building the prompt, it will perform the intent check and set the appropriate persona on the `CognitionPacket`.
*   **State Management:** The persona state will be managed on a turn-by-turn basis within the `CognitionPacket`. This is a stateless approach that is simple to implement. A more stateful session-based persona tracking could be a future enhancement if needed.
*   **Keyword-Based Intent:** The initial implementation will use a simple list of keywords. This is a pragmatic first step that can be expanded or replaced with a more sophisticated NLU model later without changing the core architecture.
