from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path

class Registry:
    """
    Simple feature registry storing vocabularies per modality and ID mapping.
    """
    def __init__(self, name: str = "default"):
        self.name = name
        self.vocabs: Dict[str, List[str]] = {}  # modality -> ordered list

    def set_vocab(self, modality: str, vocab: List[str]) -> None:
        self.vocabs[modality] = list(vocab)

    def get_vocab(self, modality: str) -> Optional[List[str]]:
        return self.vocabs.get(modality)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps({"name": self.name, "vocabs": self.vocabs}, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Registry":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        reg = Registry(name=data.get("name", "default"))
        reg.vocabs = {k: list(v) for k, v in data.get("vocabs", {}).items()}
        return reg

def intersect_vocab(v1: List[str], v2: List[str]) -> List[str]:
    s2 = set(v2)
    return [g for g in v1 if g in s2]

def align_vocabs(reg: Registry, modality: str, vocab_ref: List[str], vocab_query: List[str]) -> Tuple[List[int], List[int]]:
    """
    Return index arrays mapping aligned order for ref and query for the given modality.
    """
    aligned = intersect_vocab(vocab_ref, vocab_query)
    idx_ref = [vocab_ref.index(g) for g in aligned]
    idx_query = [vocab_query.index(g) for g in aligned]
    return idx_ref, idx_query