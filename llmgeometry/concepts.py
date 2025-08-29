"""
Concept hierarchy utilities.

Provides light-weight data structures and JSON save/load helpers for
parent/child hierarchies and their prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import json


@dataclass
class Concept:
    synset_id: str
    name: str
    prompts: List[str]


@dataclass
class ConceptHierarchy:
    parent: Concept
    children: List[Concept]
    parent_prompts: List[str]
    child_prompts: Dict[str, List[str]]  # child_id -> prompts


def save_concept_hierarchies(hierarchies: List[ConceptHierarchy], path: str) -> None:
    data = []
    for h in hierarchies:
        data.append(
            {
                "parent": asdict(h.parent),
                "children": [asdict(c) for c in h.children],
                "parent_prompts": h.parent_prompts,
                "child_prompts": h.child_prompts,
            }
        )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_concept_hierarchies(path: str) -> List[ConceptHierarchy]:
    with open(path, "r") as f:
        data = json.load(f)
    out: List[ConceptHierarchy] = []
    for item in data:
        parent = Concept(**item["parent"])
        children = [Concept(**c) for c in item.get("children", [])]
        out.append(
            ConceptHierarchy(
                parent=parent,
                children=children,
                parent_prompts=item.get("parent_prompts", []),
                child_prompts=item.get("child_prompts", {}),
            )
        )
    return out

