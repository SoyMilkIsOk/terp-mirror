"""State management for the Terp Mirror application."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from enum import Enum
from typing import Final, Optional

import time

from .prizes import GRAND_PRIZE_NAME, PrizeManager


class MirrorState(str, Enum):
    """Possible application runtime states."""

    IDLE: Final[str] = "IDLE"
    ROLLING: Final[str] = "ROLLING"
    RESULT: Final[str] = "RESULT"
    COOLDOWN: Final[str] = "COOLDOWN"


@dataclass
class MirrorStateMachine:
    """Manage state transitions and state timing."""

    roll_duration: float
    result_duration: float
    cooldown_duration: float
    dry_run: bool = False
    rng: random.Random = field(default_factory=random.Random)
    state: MirrorState = MirrorState.IDLE
    _state_started: float = field(default_factory=time.monotonic)
    current_prize: Optional[str] = None
    _pending_grand_prize: bool = False
    prize_manager: PrizeManager = field(default_factory=PrizeManager)
    last_trigger_timestamp: float = field(default_factory=time.monotonic)
    last_result_timestamp: Optional[float] = None

    def transition(self, new_state: MirrorState) -> None:
        """Transition to ``new_state`` and reset the internal timer."""

        if self.state is new_state:
            return
        self.state = new_state
        self._state_started = time.monotonic()

        if new_state is MirrorState.ROLLING:
            self.current_prize = None
            self._pending_grand_prize = False
            self.last_trigger_timestamp = self._state_started
        elif new_state is MirrorState.RESULT:
            self._resolve_result()
            self.last_result_timestamp = self._state_started

    def trigger_roll(self) -> None:
        """Start a roll sequence if the machine is idle or cooling down."""

        if self.state in {MirrorState.IDLE, MirrorState.COOLDOWN}:
            self.transition(MirrorState.ROLLING)

    def force_result(self) -> None:
        """Jump directly to the result state."""

        if self.state is not MirrorState.RESULT:
            self.transition(MirrorState.RESULT)

    def force_grand_prize(self) -> None:
        """Force the next result to be the grand prize."""

        self._pending_grand_prize = True
        self.force_result()

    def queue_manual_prize(self, prize_name: str) -> None:
        """Queue an explicit prize for the next result."""

        if prize_name == GRAND_PRIZE_NAME:
            self._pending_grand_prize = True
        else:
            self.prize_manager.queue_specific_prize(prize_name)

    def update(self) -> None:
        """Update the state based on elapsed time."""

        now = time.monotonic()
        elapsed = now - self._state_started

        if self.state is MirrorState.ROLLING and elapsed >= self.roll_duration:
            self.transition(MirrorState.RESULT)
        elif self.state is MirrorState.RESULT and elapsed >= self.result_duration:
            self.transition(MirrorState.COOLDOWN)
        elif self.state is MirrorState.COOLDOWN and elapsed >= self.cooldown_duration:
            self.transition(MirrorState.IDLE)

    def time_in_state(self) -> float:
        """Return the number of seconds elapsed in the current state."""

        return time.monotonic() - self._state_started

    def _resolve_result(self) -> None:
        """Resolve the prize outcome when entering the result state."""

        forced = self._pending_grand_prize
        if forced:
            prize_name, forced = self.prize_manager.resolve_prize(
                override=GRAND_PRIZE_NAME, forced=True
            )
        else:
            prize_name, forced = self.prize_manager.resolve_prize()

        self.current_prize = prize_name
        self._pending_grand_prize = False


