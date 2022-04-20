"""Module with all data classes about audio chunks."""

from typing import List

import tensorflow as tf

from seds_cli.seds_lib.data.audio.elements import ProductionAudioElement
from seds_cli.seds_lib.data.audio.elements import EvaluationAudioElement


def _concat_samples(samples_chunk: List[tf.Tensor]) -> tf.Tensor:
    """Concatenate samples of the chunk"""
    return tf.concat(samples_chunk, 0)


class AudioChunk:
    """AudioChunk"""


class ProductionAudioChunk:
    """ProductionAudioChunk"""

    def __init__(self, elements_chunk: List[ProductionAudioElement], num_unseen: int):
        self.elements_chunk = elements_chunk
        self.num_unseen = num_unseen
        self.received_samples_chunk = _concat_samples(
            [element.received_samples for element in self.elements_chunk]
        )


class EvaluationAudioChunk:
    """EvaluationAudioChunk"""

    def __init__(self, elements_chunk: List[EvaluationAudioElement], num_unseen: int):
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
