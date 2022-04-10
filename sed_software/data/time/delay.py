"""Delay"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Delay:
    """Dataclass about the delay."""
    inference: float
    receiving: float

    @property
    def max_delay(self):
        """Sum of all delay parts."""
        return self.inference + self.receiving
