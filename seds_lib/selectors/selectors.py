"""Module of selectors"""

from enum import Enum, auto
from typing import Type

from seds_lib.models.inference_models import BaseInferenceModel, TFLiteInferenceModel, \
    TFTensorRTModel
from seds_lib.models.saved_models import BaseSavedModel, CrnnSavedModel, \
    YamNetBaseSavedModel, YamNetExtendedSavedModel, Mono16kWaveInputSavedModel


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


class SystemModes(Enum):
    """Selector for available modes of the sound-event-detection system."""
    PRODUCTION = auto()
    EVALUATION = auto()


class LogLevels(Enum):
    """Selector for available logging levels."""
    INFO = auto()
    DEBUG = auto()
    ERROR = auto()
