"""TF saved Models"""

import logging
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_io as tfio


class BaseSavedModel(ABC):
    """ Base Model"""

    @abstractmethod
    def __init__(self, saved_model_path: str, threshold: float) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f'Loading models {self._name} from: {saved_model_path} with threshold: {threshold}'
        )
        self.saved_model_path = saved_model_path
        self.threshold = threshold

    @property
    @abstractmethod
    def _name(self) -> str:
        """name of the models"""

    @abstractmethod
    def preprocess(self, sample: tf.Tensor, data_sample_rate: int) -> tf.Tensor:
        """preprocess"""

    @abstractmethod
    def extract_prediction(self, sample_prediction: tf.Tensor) -> tuple:
        """predict"""


class Mono16kWaveInputSavedModel(BaseSavedModel, ABC):
    """Mono16kWaveInput"""

    def __init__(self, saved_model_path: str, threshold: float) -> None:
        super().__init__(saved_model_path, threshold)
        self.sample_rate = 16000

    def preprocess(self, sample: tf.Tensor, data_sample_rate: int) -> tf.Tensor:
        if not isinstance(sample, tf.Tensor) or tf.rank(sample) != 1:
            raise ValueError('Sample must be a 1D Tensor')
        if int(data_sample_rate) != self.sample_rate:
            sample = tfio.audio.resample(sample, data_sample_rate, self.sample_rate)
        return sample

    def extract_prediction(self, sample_prediction: tf.Tensor) -> tuple:
        y_pred_prob = sample_prediction[0]  # = [[y_pred_prob]] shape=(batch_size, pred)
        y_pred_label = bool(y_pred_prob > self.threshold)
        return y_pred_prob, y_pred_label


class CrnnSavedModel(Mono16kWaveInputSavedModel):
    """CrnnSavedModel"""

    @property
    def _name(self) -> str:
        return 'crnn'


class YamNetBaseSavedModel(Mono16kWaveInputSavedModel):
    """YamNetBaseSavedModel"""

    @property
    def _name(self) -> str:
        return 'yamnet-base'


class YamNetExtendedSavedModel(Mono16kWaveInputSavedModel):
    """YamNetExtendedSavedModel"""

    @property
    def _name(self) -> str:
        return 'yamnet-extended'
