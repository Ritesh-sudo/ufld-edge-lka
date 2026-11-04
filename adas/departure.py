"""Lane-departure warning (LDW) as a small state machine over per-frame lateral offsets.

A raw `abs(offset) > threshold` check flickers with detector noise. This adds the usual three
guards: a warn threshold above a (lower) clear threshold (hysteresis), a minimum number of
consecutive frames before warning (debounce), and a "lost" state when the lane estimate is
missing for too long, so a controller knows it should hand control back instead of trusting
a stale value.

States: OK -> WARN (offset beyond `warn_m` for `warn_frames` frames) -> OK (offset back inside `clear_m`),
        any -> LOST (no estimate for `lost_frames` frames) -> OK/WARN on the next valid estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    OK = 'ok'
    WARN_LEFT = 'warn_left'    # drifting toward the left line
    WARN_RIGHT = 'warn_right'
    LOST = 'lost'


@dataclass(frozen=True)
class DepartureConfig:
    # Lane 3.7 m, car ~1.8 m wide => ~0.95 m of play each side; warning at 0.6 m from center
    # leaves ~0.35 m before a wheel reaches the line.
    warn_m: float = 0.6
    clear_m: float = 0.4
    warn_frames: int = 3
    lost_frames: int = 5


class DepartureWarning:
    def __init__(self, cfg: DepartureConfig = DepartureConfig()):
        assert cfg.clear_m < cfg.warn_m
        self.cfg = cfg
        self.state = State.LOST
        self._over = 0      # consecutive frames beyond warn_m
        self._missing = 0   # consecutive frames without an estimate

    def update(self, offset_m: float | None) -> State:
        """Feed one frame's offset (+ = car right of center); None if no estimate. Returns the new state."""
        c = self.cfg
        if offset_m is None:
            self._missing += 1
            self._over = 0
            if self._missing >= c.lost_frames:
                self.state = State.LOST
            return self.state
        self._missing = 0

        in_warn = self.state in (State.WARN_LEFT, State.WARN_RIGHT)
        if abs(offset_m) > c.warn_m:
            self._over += 1
        else:
            self._over = 0

        if in_warn:
            # Hysteresis: stay in WARN until clearly back inside, and follow the drift side.
            if abs(offset_m) < c.clear_m:
                self.state = State.OK
            else:
                self.state = State.WARN_RIGHT if offset_m > 0 else State.WARN_LEFT
        elif self._over >= c.warn_frames:
            self.state = State.WARN_RIGHT if offset_m > 0 else State.WARN_LEFT
        else:
            self.state = State.OK
        return self.state
