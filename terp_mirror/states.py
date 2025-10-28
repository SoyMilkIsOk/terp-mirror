"""State management for the Terp Mirror application."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import time


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
    state: MirrorState = MirrorState.IDLE
    _state_started: float = field(default_factory=time.monotonic)

    def transition(self, new_state: MirrorState) -> None:
        """Transition to ``new_state`` and reset the internal timer."""

        if self.state is new_state:
            return
        self.state = new_state
        self._state_started = time.monotonic()

    def trigger_roll(self) -> None:
        """Start a roll sequence if the machine is idle or cooling down."""

        if self.state in {MirrorState.IDLE, MirrorState.COOLDOWN}:
            self.transition(MirrorState.ROLLING)

    def force_result(self) -> None:
        """Jump directly to the result state."""

        if self.state is not MirrorState.RESULT:
            self.transition(MirrorState.RESULT)

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

