import json
import os
import shutil
from typing import Any, Dict, Optional
from datetime import datetime


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class JSONMemory:
    """
    Lightweight persistent memory for the agent.
    Stores dataset fingerprint -> best model/metrics and notes.
    """

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {"datasets": {}, "notes": []}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup)
            self.data = {"datasets": {}, "notes": [{"ts": now_iso(), "msg": f"Memory reset; backup at {backup}"}]}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_dataset_record(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self.data.get("datasets", {}).get(fingerprint)

    def upsert_dataset_record(self, fingerprint: str, record: Dict[str, Any]) -> None:
        self.data.setdefault("datasets", {})[fingerprint] = record
        self.save()

    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()
