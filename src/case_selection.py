"""Case selection over the MIMIC-CDM dataset.

The dataset ships one hadm_id list per condition. This module flattens those
lists into (hadm_id, reference_label) pairs and offers three selection modes:
the complete 2,400-case dataset, a seeded stratified subset for quick checks,
and an explicit filter by admission id or condition.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import NamedTuple, Optional

from src.adapter import Adapter

DISEASE_ORDER: tuple[str, ...] = (
    "appendicitis", "cholecystitis", "diverticulitis", "pancreatitis",
)

SUBSET_SIZE: int = 50
SUBSET_SEED: int = 42


class CaseRef(NamedTuple):
    hadm_id: int
    reference_label: str


def _all_pairs(adapter: Adapter) -> list[CaseRef]:
    return [
        CaseRef(hadm_id=int(hadm_id), reference_label=disease)
        for disease, ids in adapter.all_hadm_ids().items()
        for hadm_id in ids
    ]


def select_cases(
    adapter: Adapter,
    cases: str = "full",
    *,
    hadm_ids: Optional[list[int]] = None,
    diseases: Optional[list[str]] = None,
    subset_size: int = SUBSET_SIZE,
    seed: int = SUBSET_SEED,
) -> list[CaseRef]:
    """Return the (hadm_id, reference_label) pairs to process.

    cases='full'    -> every case in the benchmark.
    cases='subset'  -> stratified random subset, subset_size/4 per condition.
    cases='custom'  -> the full set filtered by `hadm_ids` and/or `diseases`.
    """
    pairs = _all_pairs(adapter)

    if cases == "full":
        return pairs

    if cases == "subset":
        by_disease: dict[str, list[CaseRef]] = defaultdict(list)
        for p in pairs:
            by_disease[p.reference_label].append(p)
        rng = random.Random(seed)
        per_cell = max(1, subset_size // len(DISEASE_ORDER))
        sampled: list[CaseRef] = []
        for disease in DISEASE_ORDER:
            pool = by_disease.get(disease, [])
            if pool:
                sampled.extend(rng.sample(pool, min(per_cell, len(pool))))
        # Top up if subset_size does not divide evenly across the four cells.
        shortfall = subset_size - len(sampled)
        if shortfall > 0:
            already = {p.hadm_id for p in sampled}
            spare = [p for p in pairs if p.hadm_id not in already]
            if spare:
                sampled.extend(rng.sample(spare, min(shortfall, len(spare))))
        return sampled

    if cases == "custom":
        keep = pairs
        if diseases:
            wanted = set(diseases)
            keep = [p for p in keep if p.reference_label in wanted]
        if hadm_ids:
            wanted_ids = {int(x) for x in hadm_ids}
            keep = [p for p in keep if p.hadm_id in wanted_ids]
        return keep

    raise ValueError(f"unknown case selection mode: {cases!r}")
