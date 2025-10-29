"""Prize catalog and selection utilities for Terp Mirror."""

from __future__ import annotations

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class Prize:
    """Represent a single prize entry with a selection weight."""

    name: str
    weight: int
    description: str | None = None


@dataclass(frozen=True)
class PrizeStockConfig:
    """Runtime configuration for tracking prize stock levels."""

    track_stock: bool = False
    counts: Mapping[str, int] = field(default_factory=dict)
    grand_prize: Optional[int] = None


# Core prize catalog excluding the grand prize. Edit this list to update odds.
PRIZE_CATALOG: List[Prize] = [
    Prize("Candy + Stickers", 33, "A sweet treat and some Terp Mirror swag."),
    Prize("OG Terpscoop", 33, "A classic Terpscoop of your choosing."),
    Prize("Lighter Case", 20, "A sleek case for your lighter. Lots of designs!"),
    Prize("XL Terpscoop", 10, "An extra-large Terpscoop for big scoops."),
    Prize("Terpz T-Shirt", 3, "A comfy Terpz branded t-shirt."),
]

GRAND_PRIZE_NAME = "Grand Prize"

_ROLL_LOG_PATH = Path(__file__).resolve().parent.parent / "prize_rolls.log"
_STOCK_LOG_PATH = Path(__file__).resolve().parent.parent / "prize_stock.log"
_LOGGER = logging.getLogger("terp_mirror.prize_rolls")
_STOCK_LOGGER = logging.getLogger("terp_mirror.prize_stock")

for logger, path in ((
    _LOGGER,
    _ROLL_LOG_PATH,
), (
    _STOCK_LOGGER,
    _STOCK_LOG_PATH,
)):
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


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


def log_stock_adjustment(prize_name: str, delta: int, resulting: Optional[int]) -> None:
    """Log manual stock adjustments for auditing."""

    remaining = "∞" if resulting is None else str(resulting)
    _STOCK_LOGGER.info("%s | %+d | %s", prize_name, delta, remaining)


class PrizeManager:
    """Manage prize selection, stock tracking, and manual overrides."""

    def __init__(
        self,
        catalog: Optional[Iterable[Prize]] = None,
        *,
        rng: Optional[random.Random] = None,
        stock_config: Optional[PrizeStockConfig] = None,
        dry_run: bool = False,
    ) -> None:
        self._catalog = list(catalog) if catalog is not None else list(PRIZE_CATALOG)
        self._rng = rng or random.Random()
        self._dry_run = dry_run
        self._selected_index = 0
        self._queued_manual: Optional[str] = None
        stock_cfg = stock_config or PrizeStockConfig()
        self._track_stock = bool(stock_cfg.track_stock)
        self._stock: dict[str, Optional[int]] = {}
        if self._track_stock:
            for prize in self._catalog:
                value = stock_cfg.counts.get(prize.name)
                self._stock[prize.name] = int(value) if value is not None else 0
            if GRAND_PRIZE_NAME in stock_cfg.counts:
                self._stock[GRAND_PRIZE_NAME] = int(stock_cfg.counts[GRAND_PRIZE_NAME])
            elif stock_cfg.grand_prize is not None:
                self._stock[GRAND_PRIZE_NAME] = int(stock_cfg.grand_prize)
        self._stock.setdefault(GRAND_PRIZE_NAME, None)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def available_catalog(self) -> list[Prize]:
        if not self._track_stock:
            return list(self._catalog)
        available: list[Prize] = []
        for prize in self._catalog:
            remaining = self._stock.get(prize.name)
            if remaining is None or remaining > 0:
                available.append(prize)
        return available

    def cycle_selection(self, delta: int) -> None:
        choices = self._catalog + [Prize(GRAND_PRIZE_NAME, 0)]
        self._selected_index = (self._selected_index + delta) % len(choices)

    def current_selection(self) -> str:
        choices = self._catalog + [Prize(GRAND_PRIZE_NAME, 0)]
        return choices[self._selected_index].name

    def queue_manual_selection(self) -> Optional[str]:
        selection = self.current_selection()
        self._queued_manual = selection
        return selection

    def queue_specific_prize(self, prize_name: str) -> None:
        self._queued_manual = prize_name

    @property
    def queued_manual(self) -> Optional[str]:
        return self._queued_manual

    # ------------------------------------------------------------------
    # Stock tracking
    # ------------------------------------------------------------------
    def stock_for(self, prize_name: str) -> Optional[int]:
        return self._stock.get(prize_name)

    def adjust_stock(self, prize_name: str, delta: int) -> Optional[int]:
        if not self._track_stock:
            return None
        current = self._stock.get(prize_name, 0)
        if current is None:
            return None
        new_value = max(0, current + delta)
        self._stock[prize_name] = new_value
        log_stock_adjustment(prize_name, delta, new_value)
        return new_value

    def decrement_stock(self, prize_name: str) -> None:
        if not self._track_stock:
            return
        current = self._stock.get(prize_name)
        if current is None:
            return
        self._stock[prize_name] = max(0, current - 1)

    # ------------------------------------------------------------------
    # Prize resolution
    # ------------------------------------------------------------------
    def finalize_prize(self, prize_name: str, forced: bool) -> None:
        """Record the outcome for logging and stock management."""

        self.decrement_stock(prize_name)
        log_prize_roll(prize_name, forced=forced, dry_run=self._dry_run)

    def resolve_prize(self, override: Optional[str] = None, *, forced: bool = False) -> tuple[str, bool]:
        if override is not None:
            prize_name = override
            forced = True
        elif self._queued_manual is not None:
            prize_name = self._queued_manual
            self._queued_manual = None
            forced = True
        else:
            available = self.available_catalog()
            if not available:
                prize_name = GRAND_PRIZE_NAME
                forced = True
            else:
                prize_name = weighted_prize_choice(available, self._rng).name

        self.finalize_prize(prize_name, forced)
        return prize_name, forced


__all__ = [
    "Prize",
    "PrizeManager",
    "PrizeStockConfig",
    "PRIZE_CATALOG",
    "GRAND_PRIZE_NAME",
    "weighted_prize_choice",
    "log_prize_roll",
    "log_stock_adjustment",
]
