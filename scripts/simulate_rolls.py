"""Diagnostic script to simulate Terp Mirror prize rolls."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Iterable
import random

from terp_mirror.prizes import PRIZE_CATALOG, Prize, weighted_prize_choice


TOLERANCE_PERCENT = 5.0


def _expected_percentages(prizes: Iterable[Prize]) -> dict[str, float]:
    total_weight = sum(prize.weight for prize in prizes)
    if total_weight <= 0:
        raise ValueError("Total prize weight must be positive.")
    return {prize.name: prize.weight / total_weight * 100 for prize in prizes}


def simulate_rolls(rolls: int, seed: int | None) -> None:
    rng = random.Random(seed) if seed is not None else random.Random()
    counts: Counter[str] = Counter()

    for _ in range(rolls):
        prize = weighted_prize_choice(rng=rng)
        counts[prize.name] += 1

    expected = _expected_percentages(PRIZE_CATALOG)
    print(f"Simulated {rolls} rolls{' with seed ' + str(seed) if seed is not None else ''}.")
    print(f"Tolerance: ±{TOLERANCE_PERCENT:.1f}%")
    print()
    header = f"{'Prize':<20} {'Actual %':>10} {'Expected %':>12} {'Δ%':>8} Status"
    print(header)
    print("-" * len(header))

    within_tolerance = True
    for prize in PRIZE_CATALOG:
        actual_pct = counts[prize.name] / rolls * 100 if rolls else 0.0
        expected_pct = expected[prize.name]
        delta = actual_pct - expected_pct
        status = "OK" if abs(delta) <= TOLERANCE_PERCENT else "WARN"
        if status != "OK":
            within_tolerance = False
        print(
            f"{prize.name:<20} {actual_pct:>10.2f} {expected_pct:>12.2f} {delta:>8.2f} {status}"
        )

    print()
    if within_tolerance:
        print("All prize odds within tolerance.")
    else:
        print("One or more prizes deviated beyond tolerance.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rolls",
        type=int,
        default=1000,
        help="Number of simulated rolls to run (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )
    args = parser.parse_args(argv)
    simulate_rolls(args.rolls, args.seed)


if __name__ == "__main__":
    main()
