"""Prize catalog and selection utilities for Terp Mirror."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import logging
import random


@dataclass(frozen=True)
class Prize:
    """Represent a single prize entry with a selection weight."""

    name: str
    weight: int
    description: str | None = None


# Core prize catalog excluding the grand prize. Edit this list to update odds.
PRIZE_CATALOG: List[Prize] = [
    Prize("Sticker Pack", 35, "A bundle of limited-edition stickers."),
    Prize("Keychain", 25, "A custom Terp Wizard keychain."),
    Prize("T-Shirt", 20, "A commemorative event t-shirt."),
    Prize("Mystery Box", 15, "A surprise assortment of goodies."),
    Prize("VIP Pass", 5, "Front-row access to the next showcase."),
]

GRAND_PRIZE_NAME = "Grand Prize"

_ROLL_LOG_PATH = Path(__file__).resolve().parent.parent / "prize_rolls.log"
_LOGGER = logging.getLogger("terp_mirror.prize_rolls")
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(_ROLL_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False


def _catalog_weights(catalog: Iterable[Prize]) -> list[int]:
    return [prize.weight for prize in catalog]


def weighted_prize_choice(
    catalog: Optional[Iterable[Prize]] = None, rng: Optional[random.Random] = None
) -> Prize:
    """Return a prize using weighted random selection."""

    choices = list(catalog) if catalog is not None else PRIZE_CATALOG
    if not choices:
        raise ValueError("Prize catalog cannot be empty.")

    if any(prize.weight <= 0 for prize in choices):
        raise ValueError("All prize weights must be positive integers.")

    rng_obj = rng or random
    selection = rng_obj.choices(choices, weights=_catalog_weights(choices), k=1)
    return selection[0]


def log_prize_roll(prize_name: str, *, forced: bool, dry_run: bool) -> None:
    """Append a prize roll entry to the log file."""

    mode = "FORCED" if forced else "RANDOM"
    tag = "DRY-RUN" if dry_run else "LIVE"
    _LOGGER.info("%s | %s | %s", prize_name, mode, tag)


__all__ = [
    "Prize",
    "PRIZE_CATALOG",
    "GRAND_PRIZE_NAME",
    "weighted_prize_choice",
    "log_prize_roll",
]
