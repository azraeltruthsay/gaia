# Dev Journal - January 31, 2026

## Feature Proposal: Dynamic Status Flags and Resource-Aware "Distracted" State

### 1. Summary

This document outlines a proposal for a new feature set aimed at increasing GAIA's transparency and resource awareness. The core ideas are:

1.  **"Distracted" State**: A CPU-only "Lite" mode that GAIA enters when GPU resources are heavily utilized. This would prevent loading the "Prime" model, resulting in faster but less complex responses, reflecting a state of "distraction".
2.  **Dynamic Status Flags**: A system of flags or tags, displayed in user interfaces like Discord, that provide real-time information about GAIA's operational state.
3.  **Hydrated Cognition Packet**: A philosophy for the Cognition Packet (GCP) where it contains both machine-efficient encoded data for GAIA and human-readable "hydrated" data for user transparency.

### 2. Core Concepts

#### 2.1. The "Distracted" State and Resource-Aware Model Loading

The "Distracted" state is a direct response to the physical constraints of the system GAIA is running on.

*   **Trigger**: This state would be triggered when a monitoring component detects that GPU utilization has passed a predefined threshold (e.g., >80% for more than 5 seconds). This indicates that the GPU is busy with other tasks, such as training, running another model, or a heavy workload from another user.
*   **Behavior**:
    *   When in a "Distracted" state, the `ModelPool` will not attempt to load or use the `gpu_prime` model.
    *   All cognitive tasks, including response generation and stream observation, will fall back to the `lite` CPU model.
    *   This will result in a noticeable change in GAIA's behavior: responses will be generated faster but may be less nuanced or detailed. The back-and-forth of the `lite` model observing itself could slow down the final response time and reduce its overall effectiveness.
*   **Purpose**: This mechanism makes GAIA more resilient and responsive. Instead of waiting for a busy GPU and potentially timing out, she can provide a "best-effort" response using available resources. It also provides a naturalistic explanation for variations in her performance.

#### 2.2. Dynamic Status Flags for Transparency

To make GAIA's internal state transparent to the user, we will introduce a system of dynamic status flags. These flags will be displayed alongside her responses in supported UIs (like Discord).

*   **Proposed Flags**:
    *   **`#Status`**: Indicates GAIA's current operational state.
        *   `#Ready`: GPU is available, and `Prime` model is ready.
        *   `#Distracted`: GPU is busy; operating in `Lite` mode.
        *   `#Studying`: A training process is active.
        *   `#Observing`: Passively watching a channel without being the primary respondent.
    *   **`#Model`**: The model that generated the response.
        *   `#Prime`: The primary GPU model.
        *   `#Lite`: The CPU-based model.
        *   `#Oracle`: An external API (e.g., GPT-4, Gemini).
        *   `#Dev`: A development or test model.
    *   **`#Observer`**: The model that is observing the response stream. (e.g., `#Observer:Lite`)
    *   **`#Confidence`**: The model's confidence in its response, on a scale (e.g., `#Confidence:85%`).
    *   **`#Intent`**: The system's interpretation of the user's request (e.g., `#Intent:question`, `#Intent:command`).

*   **Example Discord Response**:
    > **GAIA** `#Ready #Prime #Confidence:95% #Intent:question`
    >
    > Heimr is a fascinating campaign setting. From my knowledge...

#### 2.3. The "Hydrated" Cognition Packet

A key concern is that adding all this metadata to the GCP will "pollute" it with tokens that are not useful for the model's reasoning process, thus reducing efficiency. The proposed solution is to treat the GCP as a "hydrated" data structure.

*   **Dual Representation**: The GCP will contain two representations of the status information:
    1.  **Machine-Readable (Encoded)**: A highly compact, semantically encoded representation for the model to use. For example, `#Status:Distracted` could be encoded as `S:D`. This is what the model would see in its prompt context.
    2.  **Human-Readable (Hydrated)**: The full, human-friendly string (e.g., `#Status:Distracted`), which is used by the UI to display the flags. This data would be passed along in the GCP's metadata but not necessarily included in the prompt itself.

*   **Semantic Encoding Cheat Sheet**: To ensure the model can efficiently interpret the encoded flags, we will provide it with a "cheat sheet" in its system prompt. This cheat sheet will define the semantic encoding. For example:
    ```
    --- STATUS CHEAT SHEET ---
    S:R = Status:Ready
    S:D = Status:Distracted
    M:P = Model:Prime
    M:L = Model:Lite
    C:95 = Confidence:95%
    ```
    This allows the model to understand the compact representation while keeping the token count low.

### 3. Technical Implementation Details

1.  **GCP (Cognition Packet) Modifications**:
    *   Add a new `Status` field to the `Header` of the `CognitionPacket`. This field will be a dictionary containing the machine-readable status flags.
    *   The `OutputRouting` section of the packet will carry the human-readable flags to the UI.

2.  **Resource Monitoring**:
    *   A background thread in `gaia-core` will be responsible for monitoring GPU utilization (e.g., using `pynvml`).
    *   This monitor will update a shared state that the `ModelPool` can access to determine if it should enter the "Distracted" state.

3.  **UI Integrations**:
    *   The `DiscordConnector` (and other UI connectors) will be updated to read the human-readable status flags from the `OutputRouting` metadata and prepend them to the response message.

### 4. Benefits

*   **Transparency**: Users will have a much clearer understanding of GAIA's internal state and why her performance may vary.
*   **Resilience**: The "Distracted" state makes GAIA more robust to resource contention.
*   **Debuggability**: The status flags will provide valuable information for debugging.
*   **Naturalism**: The concept of a "distracted" state adds a layer of naturalism to the user's interaction with GAIA.

### 5. Potential Challenges

*   **Performance Overhead**: The resource monitoring thread will add a small amount of overhead. This should be negligible.
*   **Complexity**: This feature adds complexity to the `ModelPool` and the UI connectors. This must be managed with clean, well-documented code.
*   **Token Efficiency**: Even with semantic encoding, the cheat sheet will add a small number of tokens to every prompt. The benefit of the added context is believed to outweigh this small cost.
