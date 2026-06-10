from __future__ import annotations

import random

import numpy as np
from shapely import Point



def weighted_sample_index(weights: np.ndarray, rng: random.Random) -> int:
    total = float(np.sum(weights))

    if total <= 0:
        return rng.randrange(len(weights))

    random_value = rng.random() * total
    cumulative = 0.0

    for index, weight in enumerate(weights):
        cumulative += float(weight)

        if cumulative >= random_value:
            return index

    return len(weights) - 1


def normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    clean = {
        key: max(0.0, float(value))
        for key, value in probabilities.items()
    }

    total = sum(clean.values())

    if total <= 0:
        modes = list(clean.keys())
        return {mode: 1.0 / len(modes) for mode in modes}

    return {
        key: value / total
        for key, value in clean.items()
    }


def sample_categorical(
    probabilities: dict[str, float],
    rng: random.Random,
) -> str:
    probabilities = normalize_probabilities(probabilities)

    random_value = rng.random()
    cumulative = 0.0

    for key, probability in probabilities.items():
        cumulative += probability

        if random_value <= cumulative:
            return key

    return list(probabilities.keys())[-1]


