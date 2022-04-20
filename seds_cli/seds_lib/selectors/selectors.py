"""Module of selectors"""

from enum import Enum, auto
from typing import Type

from seds_cli.seds_lib.models.inference_models import BaseInferenceModel
from seds_cli.seds_lib.models.inference_models import TFLiteInferenceModel
from seds_cli.seds_lib.models.inference_models import TFTensorRTModel
from seds_cli.seds_lib.models.saved_models import BaseSavedModel
from seds_cli.seds_lib.models.saved_models import CrnnSavedModel
from seds_cli.seds_lib.models.saved_models import YamNetBaseSavedModel
from seds_cli.seds_lib.models.saved_models import YamNetExtendedSavedModel
from seds_cli.seds_lib.models.saved_models import Mono16kWaveInputSavedModel


class AutoName(Enum):
    """Since Python 3.6"""
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class InferenceModels(Enum):
    """Inference Model Selector"""
    TFLITE = TFLiteInferenceModel
    TFTRT = TFTensorRTModel


class SavedModels(Enum):
    """Selector for saved model wrapper"""
    CRNN = CrnnSavedModel
    YAMNET_BASE = YamNetBaseSavedModel
    YAMNET_EXTENDED = YamNetExtendedSavedModel
    MONO_16K_IN = Mono16kWaveInputSavedModel


class ModelSelection:
    """Selector for a combination of inference and saved model type."""

    def __init__(self,
                 inference_model: InferenceModels,
                 saved_model: SavedModels):
        self.inference_model: Type[BaseInferenceModel] = inference_model.value
        self.saved_model: Type[BaseSavedModel] = saved_model.value


class SystemModes(AutoName):
    """Selector for available modes of the sound-event-detection system."""
    PRODUCTION = auto()
    EVALUATION = auto()


class LogLevels(AutoName):
    """Selector for available logging levels."""
    INFO = auto()
    DEBUG = auto()
    ERROR = auto()
