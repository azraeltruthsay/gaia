### 🧭 **GAIA Development Priority Matrix (Updated)**

#### 🔥 High Priority

* **CouncilManager Core System**
  *Implement Council logic, system states (Awake, Sleeping, Dreaming, Distracted, REM), dynamic model swaps, reflection deferral.*
  **Status**: In Progress
* **CLI Session for Council Deliberation**
  *Enable developer interaction with live Council dialogues, interjections, and logging.*
  **Status**: In Progress
* **ResourceMonitor Stub**
  *Check system resources (RAM/CPU) to trigger Distracted Mode.*
  **Status**: Pending
* **Graceful Shutdown Hooks**
  *Ensure session save, tree scan, embedding updates on Ctrl+C or final exit.*
* **System State Snapshot Tool**
  *Capture full, tagged + timestamped archive of GAIA instance state.*

---

#### 📊 Medium Priority

* **Refinement of Sleep/Dream/REM Transitions**
  *Finalize timers, thresholds, manual triggers for mode shifts.*
* **Reflection Queue Hooks**
  *Ensure user prompts can queue for deferred Council review in Dream/REM.*
* **Update dev\_matrix and priority\_matrix**
  *Log Council decisions, reflections, votes in structured system state files.*
* **Expose ai.status to UI**
  *Give real-time system feedback.*
* **Discord Integration**
  *Enable GAIA chat and command capabilities via Discord bot.*
* **LoRA Adapter Hook-Up**
  *Enable persona-tuned inference through LoRA adapters.*
* **TTS Integration (pyttsx3/pydub)**
  *Voice output for accessibility.*

---

#### 🌙 Lower Priority

* **Dynamic Function Reference Querying**
  *Allow GAIA to answer questions about her own methods.*
* **Initiative Loop Expanded Actions**
  *Enable more types of tasks beyond self-analysis.*
* **Persona Schema Validation**
  *Catch corrupt or invalid persona files.*
* **Command DSL Expansion**
  *Support natural language command interpretation for system tasks.*
* **WireGuard from Phone to WSL**
  *Remote shell access via phone.*

---

### 🛠️ Immediate Next Steps

1️⃣ Finalize **ResourceMonitor** stub for Distracted Mode support.
2️⃣ Draft `model_stubs.py` for Hermes, CodeMind, GAIA-Lite behavior simulation.
3️⃣ Integrate CouncilManager + CLI into main GAIA system.
4️⃣ Begin testing Council deliberations in CLI with live interjections.
