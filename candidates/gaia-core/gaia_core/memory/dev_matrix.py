import json
import os
from datetime import datetime
from typing import List, Dict
import argparse

class GAIADevMatrix:
    """
    Persistent manager for GAIA's self-development tasks.
    Stores and retrieves structured roadmap tasks.
    """

    def __init__(self, config):
        self.config = config
        self.path = os.path.join(config.system_reference_path, "dev_matrix.json")
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
            print(f"❌ Failed to save dev matrix: {e}")

    def add_task(self, label: str, purpose: str, urgency: str = "medium", impact: str = "medium", source: str = "manual") -> None:
        task = {
            "task": label,
            "purpose": purpose,
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

    def resolve_task(self, label: str) -> bool:
        for t in self.tasks:
            if t["task"] == label and t["status"] == "open":
                t["status"] = "resolved"
                t["resolved"] = datetime.utcnow().isoformat()
                self._save()
                return True
        return False

    def dump(self) -> List[Dict]:
        return self.tasks

if __name__ == "__main__":
    from gaia_core.config import Config
    config = Config()
    dev_matrix = GAIADevMatrix(config)

    parser = argparse.ArgumentParser(description="GAIA Dev Matrix CLI")
    parser.add_argument("--list", action="store_true", help="List open tasks")
    parser.add_argument("--resolve", type=str, help="Resolve a task by its label")
    args = parser.parse_args()

    if args.list:
        print(json.dumps(dev_matrix.get_open_tasks(), indent=2))
    elif args.resolve:
        if dev_matrix.resolve_task(args.resolve):
            print(f"Task '{args.resolve}' resolved.")
        else:
            print(f"Task '{args.resolve}' not found or already resolved.")