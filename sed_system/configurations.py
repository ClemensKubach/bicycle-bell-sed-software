"""Configs for all components in the Sed System"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any, Type

from sed_system import utils
from sed_system.inference_models import InferenceModels, BaseInferenceModel
from sed_system.saved_models import SavedModels, BaseSavedModel


@dataclass
class AudioConfig:
    """
    Audio config data

    Args:
        sample_rate: int
        window_length: float
        frame_length: float

    Attributes:
        sample_rate (int): Audio sampling rate.
        window_length (float): Time in seconds of the audio context used for prediction.
        frame_length (float): Minimum resolution of the time in the sliding window in seconds.
        chunk_size (int): Number of frames separating the given window.
        frame_size (int): Number of samples for the given frame.
        window_size (int): Number of samples for the given window.

    """

    sample_rate: int
    window_length: float
    frame_length: float

    def __post_init__(self):
        if self.frame_length > self.window_length:
            raise ValueError('Frame time must be shorter than the window time.')
        self.chunk_size = utils.round_up_div(self.window_length, self.frame_length)
        self.frame_size = int(self.frame_length * self.sample_rate)
        self.window_size = int(self.frame_size * self.chunk_size)


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


@dataclass(frozen=True)
class ReceiverConfig:
    """Configuration for an audio receiver."""
    audio_config: AudioConfig
    channels: int
    use_input: bool
    input_device: int
    use_output: bool
    output_device: int
    storage_size: int


@dataclass(frozen=True)
class PredictorConfig:
    """Configuration for an audio receiver."""
    audio_config: AudioConfig
    tfmodel_path: str
    model_selection: ModelSelection
    threshold: float = 0.5
    callback: Optional[Any] = None


@dataclass(frozen=True)
class SedSystemConfig:
    """Configuration for a sound-event-detection system."""
    audio_config: AudioConfig
    system_mode: SystemModes
    loglevel: LogLevels
    gpu: bool
    save_records: bool
