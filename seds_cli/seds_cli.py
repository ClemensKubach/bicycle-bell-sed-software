"""
Executable script for running the sound-event-detection system with different parametrizations.
"""
import os
from typing import Any, Optional

import fire

from seds_cli import seds_constants
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
    """CLI of the Sound Event Detection Software.

    Instead of specifying the path to a predefined saved_model in tfmodel_path,
    it can be accessed directly using !model_name (like !crnn).
    The value for saved_model will be inferred automatically, if not specified separately.

    Args:
        tfmodel_path:
            Path to the saved tensorflow model which is to be used for the execution.

    Keyword Args:
        threshold:
            Lower limit of probability for determining the presence of the
            target sound (Label True). Default at 0.5.

        channels:
            Number of channels of the audio input.
            Have to be equal the channel number of the input device.
            Default at 1 for a mono mic.

        gpu:
            Boolean whether gpu is to be used for the inference.
            Should dependent of the selected infer_model and its gpu support for the device.
            Default at False.

        infer_model:
            Selector for the type of the converted inference model strategy.
            Currently, supported modes are `tflite` and `tftrt`. Default is tflite.

        saved_model:
            Selector for the kind of model the savedModel specified in tfmodel_path is.
            Necessary for correct input/output preparation.
            The tfmodel_path model have to be one out of
            {CRNN, YAMNET_BASE, YAMNET_EXTENDED, MONO_16K_IN}.
            For MONO_16K_IN, not all features are available yet.
            It is designed for an easier integration of new models.
            Default is CRNN.

        storage_length:
            Time in seconds, which should be stored for later independent use.
            Use 0 for no storing data in memory, <0 for storing all data without upper limit.
            Use with caution, can raise an unexpected memory overflow error!

        save_records:
            Defines whether the records of the length of storage_length should be stored on disk
            at the program end.

        use_input:
            Specifies whether an input device is to be used or not.

        use_output:
            Specifies whether an output device is to be used or not.

        input_device:
            Index of the device using for input. None defines the system default device.
            Default is None.

        output_device:
            Index of the device using for output. None defines the system default device.
            Default is None.

        sample_rate:
            Value in Hz. Should correspond to the value required by the SavedModel.
            The predefined models require 16K Hz. Default at 16000.

        window_length:
            Value in seconds. A larger value means more context information which can be helpful
            for the recognition by the model. The value has only a small influence on the delay.
            Typically, as long as the sound event takes at a maximum.

        frame_length:
            Value in seconds, smaller than window_length. Smaller Values can decrease the delay
            between receiving the sound and getting a prediction result for it.
            A too small value can result in bad performance or a ValueError.

        callback:
            Function for defining a custom callback for the prediction result.
            Such a function have to follow the predefined interface:
            ```
            def name_of_my_callback(
                predictor_result: Union[ProductionPredictorResult, EvaluationPredictorResult]
            ) -> None:
                pass
                # custom code
            ```

        loglevel:
            Defines at which logging level, the logging commands should be perceived.
            Currently, supported is one of {DEBUG, INFO, ERROR}.
            Default is INFO.

        save_log:
            Defines whether the log output should be stored on disk. Default is False.
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
                 save_log: bool = False,
                 ):

        if tfmodel_path[0] == '!':
            if tfmodel_path.lower() in [f'!{model.name.lower()}' for model in list(SavedModels)]:
                selection = tfmodel_path.split('!')[1]
                # if saved_model is not specified otherwise
                if not isinstance(saved_model, str):
                    saved_model = SavedModels[selection.upper()]
                    print(saved_model)
                tfmodel_path = os.path.join(seds_constants.RES_MODELS_PATH, selection)
            else:
                raise ValueError(f'{tfmodel_path} is not a valid predefined model! '
                                 'Should be one of the for parameter saved_model mentioned models.')

        if isinstance(infer_model, str):
            infer_model = InferenceModels[infer_model.upper()]
        if isinstance(saved_model, str):
            saved_model = SavedModels[saved_model.upper()]

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
        self.save_log = save_log

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
            self.save_records,
            self.save_log,
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
