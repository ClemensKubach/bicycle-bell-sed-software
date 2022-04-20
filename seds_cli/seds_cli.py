"""
Executable script for running the sound-event-detection system with different parametrizations.
"""
from typing import Any, Optional

import fire

from seds_cli.seds_lib.data.configs.configs import AudioConfig
from seds_cli.seds_lib.data.configs.configs import SedSoftwareConfig
from seds_cli.seds_lib.data.configs.configs import PredictorConfig
from seds_cli.seds_lib.data.configs.configs import ReceiverConfig
from seds_cli.seds_lib.selectors.selectors import ModelSelection
from seds_cli.seds_lib.selectors.selectors import SystemModes
from seds_cli.seds_lib.selectors.selectors import LogLevels
from seds_cli.seds_lib.selectors.selectors import InferenceModels
from seds_cli.seds_lib.selectors.selectors import SavedModels
from seds_cli.seds_lib.software import SedSoftware


class SedsCli:
    """Highly configurable execution script
    """

    def __init__(self,
                 tfmodel_path: str,

                 threshold: float = 0.5,
                 channels: int = 1,

                 gpu: bool = False,
                 infer_model: InferenceModels = InferenceModels.TFLITE,
                 saved_model: SavedModels = SavedModels.CRNN,

                 storage_length: int = 0,
                 save_records: bool = False,

                 use_input: bool = True,
                 use_output: bool = False,

                 input_device: int = None,
                 output_device: int = None,

                 sample_rate: int = 16000,
                 window_length: float = 2.0,
                 frame_length: float = 0.001,

                 callback: Any = None,
                 loglevel: LogLevels = LogLevels.INFO,
                 ):
        self.tfmodel_path = tfmodel_path
        self.threshold = threshold
        self.channels = channels
        self.gpu = gpu
        self.infer_model = infer_model
        self.saved_model = saved_model
        self.storage_length = storage_length
        self.save_records = save_records
        self.use_input = use_input
        self.use_output = use_output
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.frame_length = frame_length
        self.callback = callback
        self.loglevel = loglevel

        # mode specific parameters
        self.mode: Optional[SystemModes] = None
        self.silent: bool = False
        self.wav_file: Optional[str] = None
        self.annotation_file: Optional[str] = None

    def production(self):
        self.mode = SystemModes.PRODUCTION
        self._execute()

    def evaluation(self,
                   wav_file: str,
                   annotation_file: str,
                   silent: bool = False,
                   ):
        if wav_file is None or annotation_file is None:
            raise FileNotFoundError('Paths to wave and corresponding csv file containing '
                                    'annotations have to be given correctly!')

        self.mode: SystemModes = SystemModes.EVALUATION
        self.silent = silent
        self.wav_file = wav_file
        self.annotation_file = annotation_file
        self._execute()

    def _execute(self):
        if self.mode is None:
            raise ValueError('Mode {production|evaluation} have to be chosen. '
                             'See --help for further details.')

        audio_config = AudioConfig(
            self.sample_rate,
            self.window_length,
            self.frame_length
        )

        system_config = SedSoftwareConfig(
            audio_config,
            self.mode,
            self.loglevel,
            self.gpu,
            self.save_records
        )
        sed = SedSoftware(system_config)

        receiver_config = ReceiverConfig(
            audio_config,
            self.channels,
            self.use_input,
            self.input_device,
            self.use_output,
            self.output_device,
            self.storage_length
        )
        if self.mode == SystemModes.PRODUCTION:
            sed.system.init_receiver(receiver_config)
        elif self.mode == SystemModes.EVALUATION:
            sed.system.init_receiver(
                receiver_config,
                self.wav_file,
                self.annotation_file,
                self.silent
            )

        selected_model = ModelSelection(
            self.infer_model,
            self.saved_model
        )
        predictor_config = PredictorConfig(
            audio_config,
            self.tfmodel_path,
            selected_model,
            self.threshold,
            self.callback
        )
        sed.system.init_predictor(predictor_config)

        sed.start()


def main():
    fire.Fire(SedsCli)


if __name__ == '__main__':
    main()
