"""Module with all data classes about component configurations."""

from dataclasses import dataclass
from typing import Optional, Any

from seds_cli.seds_lib.selectors.selectors import ModelSelection
from seds_cli.seds_lib.selectors.selectors import SystemModes
from seds_cli.seds_lib.selectors.selectors import LogLevels
from seds_cli.seds_lib.utils.audio_maths import round_up_div


@dataclass
class AudioConfig:
    """Contains settings concerning the audio data.

    Args
        sample_rate: int
            Value in Hz. Typically, 16000.
        window_length: float
            Value in seconds. A larger value means more context information which can be helpful
            for the recognition by the model. The value has only a small influence on the delay.
            Typically, as long as the sound event takes at a maximum.
        frame_length: float
            Value in seconds, smaller than window_length. Smaller Values can decrease the delay
            between receiving the sound and getting a prediction result for it.
            A too small value can result in bad performance or a ValueError.

    Raises
        ValueError

    Attributes
        sample_rate (int)
            Audio sampling rate in Hz
        window_length (float)
            Time in seconds of the audio context used for prediction.
        frame_length (float)
            Minimum resolution of the time in the sliding window in seconds.
        chunk_size (int)
            Number of frames separating the given window.
        frame_size (int)
            Number of samples for the given frame.
        window_size (int)
            Number of samples for the given window.
    """

    sample_rate: int
    window_length: float
    frame_length: float

    def __post_init__(self):
        if self.frame_length > self.window_length:
            raise ValueError('Frame time must be shorter than the window time.')
        self.chunk_size = round_up_div(self.window_length, self.frame_length)
        self.frame_size = int(self.frame_length * self.sample_rate)
        if self.frame_size < 1:
            raise ValueError('Frame time must be larger such that a frame contains at '
                             'least frame_length*sample_rate>=1 sample points.')
        self.window_size = int(self.frame_size * self.chunk_size)


@dataclass(frozen=True)
class ReceiverConfig:
    """Contains settings concerning the Receiver.

    Attributes
        audio_config (AudioConfig)
            Audio config object.
        channels (int)
            Number of channels of the audio input.
            Have to be equal the channel number of the input device.
            Typically, 1 for mono mic or 2 for a stereo device.
        use_input (bool)
            Specifies whether an input device is to be used or not.
        input_device (int)
            Index of the device using for input.
        use_output (bool)
            Specifies whether an output device is to be used or not.
        output_device (int)
            Index of the device using for output.
        storage_length (float)
            Time in seconds, which should be stored for later independent use.
            Use 0 for no storing data in memory, <0 for storing all data without upper limit.
            Use with caution, can raise an unexpected memory overflow error!
    """

    audio_config: AudioConfig
    channels: int
    use_input: bool
    input_device: int
    use_output: bool
    output_device: int
    storage_length: float


@dataclass(frozen=True)
class PredictorConfig:
    """Contains settings concerning the Predictor.

    Attributes
        audio_config (AudioConfig)
            Audio config object.
        tfmodel_path (str)
            Path to the tensorflow SavedModel.
        model_selection (ModelSelection)
            Choose a mixture of model building for inference.
        threshold (float)
            Value between 0 and 1, as lower bound of the precision probability
            for accepting the corresponding sound as the target sound event. Default is 0.5.
        callback (func)
            A function used for customize what should be done with each prediction result.
            With the following signature:
            def my_custom_callback(predictor_result) -> None
            Where predictor_result is of type ProductionPredictorResult or
            EvaluationPredictorResult.
            Should be kept fast to keep the predictor thread and its delay low.
            If None, a default logging callback will be applied.
    """
    audio_config: AudioConfig
    tfmodel_path: str
    model_selection: ModelSelection
    threshold: float = 0.5
    callback: Optional[Any] = None


@dataclass(frozen=True)
class SedSoftwareConfig:
    """Contains settings concerning the sed software.

    Attributes
        audio_config (AudioConfig)
            Audio config object.
        system_mode (SystemModes)
            Defines in which mode or for which purpose the system will be started.
        loglevel (LogLevels)
            Defines at which level messages should be logged.
        gpu (bool)
            Defines whether the gpu of the machine should be used.
        save_records (bool)
            Defines whether the records of the length of storage_length should be stored on disk
            at the program end.
        save_log (bool)
            Defines whether the log output should be stored on disk.
    """
    audio_config: AudioConfig
    system_mode: SystemModes
    loglevel: LogLevels
    gpu: bool
    save_records: bool
    save_log: bool
