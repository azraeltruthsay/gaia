import json
import os
from typing import List, Dict, Optional
from datetime import datetime

class GAIAPriorityManager:
    """
    Centralized persistent task and priority memory for GAIA.
    Tracks open tasks from all sources (manual, reflection, logs, EXECUTE failures).
    """

    def __init__(self, config):
        self.config = config
        self.path = os.path.join(config.reflections_path, "priority_matrix.json")
        self.tasks: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.tasks = json.load(f)
            except Exception:
                self.tasks = []

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.tasks, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save priority matrix: {e}")

    def add_task(self, label: str, details: str, urgency: str = "medium", impact: str = "medium", source: str = "manual") -> None:
        task = {
            "task": label,
            "details": details,
            "urgency": urgency,
            "impact": impact,
            "source": source,
            "status": "open",
            "created": datetime.utcnow().isoformat()
        }
        self.tasks.append(task)
        self._save()

    def get_open_tasks(self) -> List[Dict]:
        return [t for t in self.tasks if t.get("status") == "open"]

    def get_top_tasks(self, limit: int = 5) -> List[Dict]:
        def score(t):
            return (
                {"low": 1, "medium": 2, "high": 3}.get(t["urgency"], 2) +
                {"low": 1, "medium": 2, "high": 3}.get(t["impact"], 2)
            )
        return sorted(self.get_open_tasks(), key=score, reverse=True)[:limit]

    def resolve_task(self, label: str) -> bool:
        for t in self.tasks:
            if t["task"] == label and t["status"] == "open":
                t["status"] = "resolved"
                t["resolved"] = datetime.utcnow().isoformat()
                self._save()
                return True
        return False

    def clear_resolved(self):
        self.tasks = [t for t in self.tasks if t.get("status") != "resolved"]
        self._save()

    def dump(self) -> List[Dict]:
        return self.tasks
