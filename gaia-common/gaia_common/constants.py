from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class GaiaConstants:
    LOGICAL_STOP_PUNCTUATION: List[str] = field(default_factory=lambda: [".", "!", "?", "
"])
    MAX_ALLOWED_RESPONSE_TOKENS: int = 2048
    RESPONSE_BUFFER: int = 768

@dataclass
class TokenBudgets:
    full: int = 8192
    medium: int = 4096
    minimal: int = 2048

@dataclass
class Fragmentation:
    enabled: bool = True
    continuation_threshold: float = 0.85
    max_fragments: int = 5
    verification_enabled: bool = True
    seam_overlap_tokens: int = 20

@dataclass
class ToolRouting:
    ENABLED: bool = True
    SELECTION_TEMPERATURE: float = 0.15
    REVIEW_TEMPERATURE: float = 0.3
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_REINJECTIONS: int = 3
    ALLOW_WRITE_TOOLS: bool = False
    ALLOW_EXECUTE_TOOLS: bool = False

@dataclass
class AllConstants:
    GAIA: GaiaConstants = field(default_factory=GaiaConstants)
    TOKEN_BUDGETS: TokenBudgets = field(default_factory=TokenBudgets)
    FRAGMENTATION: Fragmentation = field(default_factory=Fragmentation)
    TOOL_ROUTING: ToolRouting = field(default_factory=ToolRouting)

constants = AllConstants()
