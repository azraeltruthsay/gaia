# gaia-monkey — The Adversary

**Port:** 6420 | **GPU:** No | **Dependencies:** FastAPI, docker.sock (ro)

gaia-monkey is the adversarial chaos testing service. It deliberately breaks things to prove GAIA can recover.

## Design Principles

- **Separated from immune system**: Active testing (monkey) vs passive monitoring (doctor)
- **Defensive Meditation**: Time-boxed window (30 min) where chaos drills are permitted
- **Serenity State**: Trust signal accumulated through successful recovery — gates autonomous promotion

## Chaos Drill Types

| Type | Endpoint | What It Does |
|------|----------|-------------|
| Container | `/chaos/drill` | Stops containers, verifies failure, restarts, validates recovery |
| Code | `/chaos/code` | Injects semantic faults in .py files, sends to Core for LLM repair |
| Linguistic | `/chaos/linguistic` | PromptFoo red-team evaluation (persona, factuality, format) |

## Serenity Scoring

Points are only awarded for **LLM-repaired** faults, proving the full cognitive pipeline works — not just container restart. Serenity state is shared via `/shared/doctor/serenity.json`.

## Scheduler Modes

- **triggered**: Runs when manually triggered
- **scheduled**: Fixed interval between drills
- **random**: Re-randomized after each run
- **persistent**: Cooldown loop
