"""Audio elements"""

from abc import ABC
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class AudioElement(ABC):
    """AudioElement"""

    received_samples: tf.Tensor


@dataclass
class ProductionAudioElement(AudioElement):
    """ProductionAudioElement"""


@dataclass
class EvaluationAudioElement(AudioElement):
    """EvaluationAudioElement"""

    played_samples: tf.Tensor
    labels: tf.Tensor

    def __post_init__(self):
        if not (self.received_samples.shape == self.played_samples.shape and
                self.received_samples.shape == self.labels.shape):
            raise ValueError("Shapes of received_samples, played_samples and labels are not equal")
