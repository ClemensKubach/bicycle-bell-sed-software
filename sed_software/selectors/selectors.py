"""Configs for all components in the Sed System"""

from enum import Enum, auto
from typing import Type

from sed_software.models.inference_models import InferenceModels, BaseInferenceModel
from sed_software.models.saved_models import SavedModels, BaseSavedModel


class ModelSelection:
    """Select combination of inference and saved model"""

    def __init__(self,
                 inference_model: InferenceModels,
                 saved_model: SavedModels):
        self.inference_model: Type[BaseInferenceModel] = inference_model.value
        self.saved_model: Type[BaseSavedModel] = saved_model.value


class SystemModes(Enum):
    """Available modes for the sound-event-detection system."""
    PRODUCTION = auto()
    EVALUATION = auto()


class LogLevels(Enum):
    """Available logging levels."""
    INFO = auto()
    DEBUG = auto()
    ERROR = auto()
