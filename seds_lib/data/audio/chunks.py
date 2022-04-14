"""Module with all data classes about audio chunks."""

from typing import Protocol

import tensorflow as tf

from seds_lib.data.audio.elements import ProductionAudioElement, EvaluationAudioElement


def _concat_samples(samples_chunk: list[tf.Tensor]) -> tf.Tensor:
    """Concatenate samples of the chunk"""
    return tf.concat(samples_chunk, 0)


class AudioChunk(Protocol):
    """AudioChunk"""


class ProductionAudioChunk:
    """ProductionAudioChunk"""

    def __init__(self, elements_chunk: list[ProductionAudioElement], num_unseen: int):
        self.elements_chunk = elements_chunk
        self.num_unseen = num_unseen
        self.received_samples_chunk = _concat_samples(
            [element.received_samples for element in self.elements_chunk]
        )


class EvaluationAudioChunk:
    """EvaluationAudioChunk"""

    def __init__(self, elements_chunk: list[EvaluationAudioElement], num_unseen: int):
        self.elements_chunk = elements_chunk
        self.num_unseen = num_unseen
        self.received_samples_chunk = _concat_samples(
            [element.received_samples for element in self.elements_chunk]
        )
        self.played_samples_chunk = _concat_samples(
            [element.played_samples for element in self.elements_chunk]
        )
        self.labels_chunk = _concat_samples(
            [element.labels for element in self.elements_chunk]
        )
