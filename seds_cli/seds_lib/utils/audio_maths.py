"""Module with utils functions for audio related calculations."""

from typing import Union


def samples_to_seconds(samples: Union[int, list], sample_rate: int) -> float:
    """return time in seconds of sample size or samples list"""
    if isinstance(samples, list):
        length = len(samples)
    else:
        length = samples
    return length / sample_rate


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """return number of samples for a time in seconds. Uses int() rounding."""
    return int(seconds * sample_rate)


def round_up_div(num_a, num_b) -> int:
    """division with round-up(num_a // num_b)"""
    return int(num_a // num_b + (num_a % num_b > 0))
