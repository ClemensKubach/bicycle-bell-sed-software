"""Module containing supported models (SavedModel) with their requirements and
interface for input and output."""

import logging
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_io as tfio


class BaseSavedModel(ABC):
    """Abstract base model for all saved models."""

    @abstractmethod
    def __init__(self, saved_model_path: str, threshold: float) -> None:
        """Expects an absolut path to a supported TensorFlow SavedModel and a threshold value for
        label determination from the prediction probability value."""
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f'Loading models {self._name} from: {saved_model_path} with threshold: {threshold}'
        )
        self.saved_model_path = saved_model_path
        self.threshold = threshold

    @property
    @abstractmethod
    def _name(self) -> str:
        """name of the model just for logging"""

    @abstractmethod
    def preprocess(self, sample: tf.Tensor, data_sample_rate: int) -> tf.Tensor:
        """Defines the preprocessing step for fitting the input format requirements."""

    @abstractmethod
    def extract_prediction(self, sample_prediction: float) -> tuple:
        """Defines how the expected output results can be extracted from the output format of the
        inference model. The threshold value is used for label determination.

        Returns
            tuple[prob, label] where prob is a single float and label is a single bool.
        """


class Mono16kWaveInputSavedModel(BaseSavedModel):
    """Abstract base class for all saved models expecting a mono wave file
    with a 16kHz sampling rate as input
    """

    @property
    def _name(self) -> str:
        return 'Mono16kWaveInputModel'

    def __init__(self, saved_model_path: str, threshold: float) -> None:
        super().__init__(saved_model_path, threshold)
        self.sample_rate = 16000

    def preprocess(self, sample: tf.Tensor, data_sample_rate: int) -> tf.Tensor:
        if not isinstance(sample, tf.Tensor) or tf.rank(sample) != 1:
            raise ValueError('Sample must be a 1D Tensor')
        if int(data_sample_rate) != int(self.sample_rate):
            sample = tfio.audio.resample(sample, data_sample_rate, self.sample_rate)
            # if tf bug about necessary (unused) tensorflow_io import is resolved, use resampy:
            # sample = resampy.resample(sample_np_array, data_sample_rate, self.sample_rate)
        return sample

    def extract_prediction(self, sample_prediction: float) -> tuple:
        y_pred_prob = sample_prediction
        y_pred_label = bool(y_pred_prob > self.threshold)
        return y_pred_prob, y_pred_label


class CrnnSavedModel(Mono16kWaveInputSavedModel):
    """Wrapper for a direct use of the CRNN saved under the resource folder 'models'.
    Is mentioned in SavedModels selector to make the usage more comprehensible.

    Expects the I/O formats defined in parent class Mono16kWaveInputSavedModel.

    Note for Future Development
        Should be deleted for a later even more independent and modular system.
        Instead of this, Mono16kWaveInputSavedModel should be used directly in the selector.
    """

    @property
    def _name(self) -> str:
        return 'crnn'


class YamNetBaseSavedModel(Mono16kWaveInputSavedModel):
    """Wrapper for a direct use of the base YamNet model saved under the resource folder 'models'.
    Is mentioned in SavedModels selector to make the usage more comprehensible.

    Expects the I/O formats defined in parent class Mono16kWaveInputSavedModel.

    Note for Future Development
        Should be deleted for a later even more independent and modular system.
        Instead of this, Mono16kWaveInputSavedModel should be used directly in the selector.
    """

    @property
    def _name(self) -> str:
        return 'yamnet-base'


class YamNetExtendedSavedModel(Mono16kWaveInputSavedModel):
    """Wrapper for a direct use of the extended YamNet model saved under
    the resource folder 'models'.
    Is mentioned in SavedModels selector to make the usage more comprehensible.

    Expects the I/O formats defined in parent class Mono16kWaveInputSavedModel.

    Note for Future Development
        Should be deleted for a later even more independent and modular system.
        Instead of this, Mono16kWaveInputSavedModel should be used directly in the selector.
    """

    @property
    def _name(self) -> str:
        return 'yamnet-extended'
