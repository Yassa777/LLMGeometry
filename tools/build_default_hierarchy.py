#!/usr/bin/env python3
"""
Write a curated, multi-domain concept hierarchy JSON suitable for LLMGeometry.

Parents span diverse domains (animals, vehicles, professions, foods, emotions,
programming, geography, arts). Each child has several prompts. This is meant to
be a solid starting dataset; you can edit/extend as needed.

Usage:
  python tools/build_default_hierarchy.py --out runs/exp01/concept_hierarchies.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Write a curated concept hierarchy JSON")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    def parent(name, synset, parent_prompts, children):
        return {
            "parent": {"synset_id": synset, "name": name, "prompts": []},
            "children": [{"synset_id": c[0], "name": c[1], "prompts": []} for c in children],
            "parent_prompts": parent_prompts,
            "child_prompts": {},
        }

    data = []

    # Animals
    animals = parent(
        "animal",
        "p_animal",
        [
            "A sentence about an animal in the wild.",
            "Describe an animal's behavior.",
            "A short fact about animals.",
            "An observation about animal habitats.",
            "A sentence about animal offspring.",
            "A sentence about animal adaptation.",
            "A sentence about animal diet.",
            "A sentence about animal locomotion.",
        ],
        [("c_dog", "dog"), ("c_cat", "cat"), ("c_bird", "bird"), ("c_horse", "horse")],
    )
    animals["child_prompts"] = {
        "c_dog": [
            "A sentence about a dog.",
            "Describe a dog's behavior.",
            "A short fact about dogs.",
            "A sentence about a puppy.",
        ],
        "c_cat": [
            "A sentence about a cat.",
            "Describe a cat's behavior.",
            "A short fact about cats.",
            "A sentence about a kitten.",
        ],
        "c_bird": [
            "A sentence about a bird.",
            "Describe a bird's flight.",
            "A short fact about birds.",
            "A sentence about bird songs.",
        ],
        "c_horse": [
            "A sentence about a horse.",
            "Describe a horse's gait.",
            "A short fact about horses.",
            "A sentence about foals.",
        ],
    }
    data.append(animals)

    # Vehicles
    vehicles = parent(
        "vehicle",
        "p_vehicle",
        [
            "A sentence about a form of transport.",
            "Describe a vehicle's purpose.",
            "A short fact about vehicles.",
            "A sentence about vehicle safety.",
        ],
        [("c_car", "car"), ("c_airplane", "airplane"), ("c_bicycle", "bicycle"), ("c_boat", "boat")],
    )
    vehicles["child_prompts"] = {
        "c_car": [
            "A sentence about a car.",
            "Describe car fuel efficiency.",
            "A short fact about cars.",
        ],
        "c_airplane": [
            "A sentence about an airplane.",
            "Describe a plane's takeoff.",
            "A short fact about airplanes.",
        ],
        "c_bicycle": [
            "A sentence about a bicycle.",
            "Describe a bike's gears.",
            "A short fact about bicycles.",
        ],
        "c_boat": [
            "A sentence about a boat.",
            "Describe a boat on water.",
            "A short fact about boats.",
        ],
    }
    data.append(vehicles)

    # Professions
    professions = parent(
        "profession",
        "p_profession",
        [
            "A sentence about a profession.",
            "Describe a professional's duties.",
            "A short fact about careers.",
        ],
        [("c_doctor", "doctor"), ("c_lawyer", "lawyer"), ("c_engineer", "engineer"), ("c_teacher", "teacher")],
    )
    professions["child_prompts"] = {
        "c_doctor": [
            "A sentence about a doctor.",
            "Describe a doctor's work.",
            "A short fact about medicine.",
        ],
        "c_lawyer": [
            "A sentence about a lawyer.",
            "Describe a lawyer's duties.",
            "A short fact about law.",
        ],
        "c_engineer": [
            "A sentence about an engineer.",
            "Describe an engineer's project.",
            "A short fact about engineering.",
        ],
        "c_teacher": [
            "A sentence about a teacher.",
            "Describe a teacher's classroom.",
            "A short fact about education.",
        ],
    }
    data.append(professions)

    # Foods
    foods = parent(
        "food",
        "p_food",
        [
            "A sentence about cuisine.",
            "Describe a meal.",
            "A short fact about food.",
        ],
        [("c_pizza", "pizza"), ("c_sushi", "sushi"), ("c_salad", "salad"), ("c_burger", "burger")],
    )
    foods["child_prompts"] = {
        "c_pizza": ["A sentence about pizza.", "Describe a pizza topping.", "A short fact about pizza."],
        "c_sushi": ["A sentence about sushi.", "Describe sushi rice.", "A short fact about sushi."],
        "c_salad": ["A sentence about a salad.", "Describe a salad dressing.", "A short fact about salads."],
        "c_burger": ["A sentence about a burger.", "Describe a burger patty.", "A short fact about burgers."],
    }
    data.append(foods)

    # Emotions
    emotions = parent(
        "emotion",
        "p_emotion",
        [
            "A sentence about emotions.",
            "Describe how emotions influence decisions.",
            "A short fact about human feelings.",
        ],
        [("c_happiness", "happiness"), ("c_anger", "anger"), ("c_sadness", "sadness"), ("c_surprise", "surprise")],
    )
    emotions["child_prompts"] = {
        "c_happiness": ["A sentence about happiness.", "Describe what brings joy.", "A short fact about happiness."],
        "c_anger": ["A sentence about anger.", "Describe a source of anger.", "A short fact about anger."],
        "c_sadness": ["A sentence about sadness.", "Describe coping with sadness.", "A short fact about sadness."],
        "c_surprise": ["A sentence about surprise.", "Describe being surprised.", "A short fact about surprise."],
    }
    data.append(emotions)

    # Programming
    prog = parent(
        "programming language",
        "p_programming",
        [
            "A sentence about programming languages.",
            "Describe a compiler or interpreter.",
            "A short fact about code.",
        ],
        [("c_python", "python"), ("c_javascript", "javascript"), ("c_java", "java"), ("c_cpp", "c++")],
    )
    prog["child_prompts"] = {
        "c_python": ["A sentence about Python.", "Describe Python packages.", "A short fact about Python."],
        "c_javascript": ["A sentence about JavaScript.", "Describe JS in a browser.", "A short fact about JS."],
        "c_java": ["A sentence about Java.", "Describe the JVM.", "A short fact about Java."],
        "c_cpp": ["A sentence about C++.", "Describe templates in C++.", "A short fact about C++."],
    }
    data.append(prog)

    # Geography
    geo = parent(
        "geography",
        "p_geography",
        [
            "A sentence about landscapes.",
            "Describe a natural region.",
            "A short fact about geography.",
        ],
        [("c_mountain", "mountain"), ("c_river", "river"), ("c_desert", "desert"), ("c_forest", "forest")],
    )
    geo["child_prompts"] = {
        "c_mountain": ["A sentence about a mountain.", "Describe mountain climate.", "A short fact about mountains."],
        "c_river": ["A sentence about a river.", "Describe a river's flow.", "A short fact about rivers."],
        "c_desert": ["A sentence about a desert.", "Describe desert conditions.", "A short fact about deserts."],
        "c_forest": ["A sentence about a forest.", "Describe forest ecology.", "A short fact about forests."],
    }
    data.append(geo)

    # Arts
    arts = parent(
        "art",
        "p_arts",
        [
            "A sentence about the arts.",
            "Describe an art form.",
            "A short fact about art history.",
        ],
        [("c_painting", "painting"), ("c_sculpture", "sculpture"), ("c_music", "music"), ("c_dance", "dance")],
    )
    arts["child_prompts"] = {
        "c_painting": ["A sentence about painting.", "Describe paint on canvas.", "A short fact about painting."],
        "c_sculpture": ["A sentence about sculpture.", "Describe a sculpture material.", "A short fact about sculpture."],
        "c_music": ["A sentence about music.", "Describe a musical instrument.", "A short fact about music."],
        "c_dance": ["A sentence about dance.", "Describe choreography.", "A short fact about dance."],
    }
    data.append(arts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Wrote curated hierarchy:", out_path)


if __name__ == "__main__":
    main()

