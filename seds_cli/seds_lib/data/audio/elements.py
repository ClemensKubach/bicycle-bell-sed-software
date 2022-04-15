"""Module with all data classes about audio elements."""

from abc import ABC
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class AudioElement(ABC):
    """Abstract base class of an audio element.
    An audio element is a mini-batch of sample points and corresponding data.

    It contains tensors with the individual audio sample data points.
    """

    received_samples: tf.Tensor


@dataclass
class ProductionAudioElement(AudioElement):
    """AudioElement for the production mode.
    An audio element is a mini-batch of sample points and corresponding data.
    Typically, received by the ProductionAudioReceiver as frame for each callback.

    Attributes

    received_samples (tf.Tensor)
        of shape (num_samples,) where num_samples equals frame_size
    """


@dataclass
class EvaluationAudioElement(AudioElement):
    """AudioElement for the evaluation mode.
    An audio element is a mini-batch of sample points and corresponding data.
    Typically, received by the EvaluationAudioReceiver as frame for each callback.

    Attributes

    received_samples (tf.Tensor)
        of shape (num_samples,) where num_samples equals frame_size

    played_samples (tf.Tensor)
        of shape (num_samples,) where num_samples equals frame_size

    labels (tf.Tensor)
        of shape (num_samples,) where num_samples equals frame_size
    """

    played_samples: tf.Tensor
    labels: tf.Tensor

    def __post_init__(self):
        if not (self.received_samples.shape == self.played_samples.shape and
                self.received_samples.shape == self.labels.shape):
            raise ValueError(
                "Shapes of received_samples, played_samples and labels have to be equal"
            )
