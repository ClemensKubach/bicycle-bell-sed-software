"""
Executable script for running the sound-event-detection system with different parametrizations.
"""
import logging
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
from seds_cli.seds_lib.utils import audio_maths, file_utils


class SedsCli:
    """CLI of the Sound Event Detection Software."""

    # noinspection PyPep8Naming
    # pylint:disable=invalid-name
    # pylint:disable=no-self-use
    class resources:
        """Cli for resources related actions.

        Actions:
            where
        """

        # noinspection PyMethodMayBeStatic
        def where(self):
            """Print base dir of resource folder"""
            logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
            logging.root.setLevel(logging.INFO)
            _logger = logging.getLogger(__name__)
            _logger.info(f'Resource files are located at: {seds_constants.RES_PATH}')

    # noinspection PyPep8Naming
    # pylint:disable=invalid-name
    class devices:
        """Cli for device related actions.

        Actions:
            soundcheck
        """

        # noinspection PyMethodMayBeStatic
        # pylint:disable=import-outside-toplevel
        def soundcheck(self,
                       input_device: int = None,
                       output_device: int = None,
                       channels: int = 2,
                       ):
            """Runs a playback soundcheck with the chosen devices.

            Args:
                input_device:
                    Index (int) of the device using for input.
                    None defines the system default device.

                output_device:
                    Index (int) of the device using for output.
                    None defines the system default device.

                channels:
                    Number of channels of the audio input.
                    Have to be equal the channel number of the input device.
                    Default at 2 for a stereo mic.
            """
            import pyaudio
            import time
            p = pyaudio.PyAudio()
            logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
            logging.root.setLevel(logging.INFO)
            _logger = logging.getLogger(__name__)

            # pylint:disable=unused-argument
            def callback(in_data, frame_count, time_info, status):
                return in_data, pyaudio.paContinue

            _logger.info('Start to talk or play a sound out loud '
                         'to receive the playback via speaker...')

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=16000,
                input=True,
                output=True,
                input_device_index=input_device,
                output_device_index=output_device,
                stream_callback=callback,
            )

            stream.start_stream()
            try:
                while stream.is_active():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                _logger.info('Stopped playback')
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

    # noinspection PyPep8Naming
    # pylint:disable=invalid-name
    # pylint:disable=too-many-instance-attributes
    class run:
        """Cli to start a Sound Event Detection.

        Actions:
            production

            evaluation

        Usage:
            Default channel number is set to 2 for a stereo mic.

            Instead of specifying the path to a predefined saved_model in `tfmodel`,
            it can be accessed directly using !model_name (like !crnn).
            The value for saved_model will be inferred automatically, if not specified separately.

        Functionality:
            There are 2 modes to run the system. The **Production** and the **Evaluation** mode.

            With the **Production** mode, the sounds of the environment are recorded and evaluated
            with the help of the selected microphone.

            In the **Evaluation** mode, the system can be tested using a selected wave file and
            an associated CSV file, containing start and end time of the contiguous presence of the
            target sound event in each line. The output contains an indication of the annotated
            ground-truth value from the CSV,
            a prediction value for the played wave data and a prediction value for the recorded
            audio. If `silent=True`, it is no audio played out loud.

        Args:
            tfmodel:
                Path to the saved tensorflow model which is to be used for the execution.

        Keyword Args:
            threshold:
                Lower limit of probability for determining the presence of the
                target sound (Label True).

            channels:
                Number of channels of the audio input.
                Have to be equal the channel number of the input device.
                Default at 2 for a stereo mic.

            gpu:
                Boolean whether gpu is to be used for the inference.
                Should dependent of the selected infer_model and its gpu support for the device.

            infer_model:
                Selector for the type of the converted inference model strategy.
                Currently, supported modes are `tflite` and `tftrt`.

            saved_model:
                Selector for the kind of model the savedModel specified in tfmodel is.
                Necessary for correct input/output preparation.
                The tfmodel model have to be one out of
                {CRNN, YAMNET_BASE, YAMNET_EXTENDED, MONO_16K_IN}.
                For MONO_16K_IN, not all features are available yet.
                It is designed for an easier integration of new models.

            storage_length:
                Time in seconds, which should be stored for later independent use.
                Use 0 for no storing data in memory, <0 for storing all data without upper limit.
                Use with caution, can raise an unexpected memory overflow error!

            save_records:
                Defines whether the records of the length of storage_length should be stored on disk
                at the program end.

            input_device:
                Index of the device using for input. None defines the system default device.

            output_device:
                Index of the device using for output. None defines the system default device.

            sample_rate:
                Value in Hz. Should correspond to the value required by the SavedModel.
                The predefined models require 16K Hz.

            window_length:
                Value in seconds. A larger value means more context information which can be helpful
                for the recognition by the model.
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

            save_log:
                Defines whether the log output should be stored on disk.
        """

        # pylint:disable=too-many-arguments
        # pylint:disable=too-many-locals
        def __init__(self,
                     tfmodel: str,

                     threshold: float = 0.5,
                     channels: int = 2,

                     gpu: bool = False,
                     infer_model: InferenceModels = InferenceModels.TFLITE,
                     saved_model: SavedModels = SavedModels.CRNN,

                     storage_length: int = 0,
                     save_records: bool = False,

                     input_device: int = None,
                     output_device: int = None,

                     sample_rate: int = 16000,
                     window_length: float = 1.0,
                     frame_length: float = 0.001,

                     callback: Any = None,
                     loglevel: LogLevels = LogLevels.INFO,
                     save_log: bool = False,
                     ):

            if tfmodel is None:
                ValueError("Path to TF SavedModel or one of the predefined model shortcuts "
                           "(like !crnn) must be given!")

            if tfmodel[0] == '!':
                if tfmodel.lower() in [f'!{model.name.lower()}' for model in
                                       list(SavedModels)]:
                    selection = tfmodel.split('!')[1]
                    # if saved_model is not specified otherwise
                    if not isinstance(saved_model, str):
                        saved_model = SavedModels[selection.upper()]
                    tfmodel = os.path.join(seds_constants.RES_MODELS_PATH, selection)
                else:
                    raise ValueError(f'{tfmodel} is not a valid predefined model! Should be '
                                     'one of the for parameter saved_model mentioned models.')

            if isinstance(infer_model, str):
                infer_model = InferenceModels[infer_model.upper()]
            if isinstance(saved_model, str):
                saved_model = SavedModels[saved_model.upper()]

            self.tfmodel_path = tfmodel
            self.threshold = threshold
            self.channels = channels
            self.gpu = gpu
            self.infer_model = infer_model
            self.saved_model = saved_model
            self.storage_length = storage_length
            self.save_records = save_records
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
            self.use_input: bool = True
            self.use_output: bool = False
            self.wav_file: Optional[str] = None
            self.annotation_file: Optional[str] = None

            print(f'Parameters: {self}')

        def __str__(self):
            object_dict = self.__dict__
            return str(object_dict)

        def production(self,
                       use_output: bool = False,
                       ):
            """run system in production mode

            Keyword Args:
                use_output:
                    Specifies whether an output device is to be used or not (for playback).

            """
            self.mode = SystemModes.PRODUCTION
            self.use_output = use_output
            self._execute()

        def evaluation(self,
                       wav_file: Optional[str] = None,
                       annotation_file: Optional[str] = None,
                       silent: bool = False,
                       use_input: bool = True,
                       use_output: bool = True,
                       ):
            """run system in evaluation mode

            Keyword Args:
                wav_file:
                    Path to the wave file which shall be used.
                    If None, the predefined mini-test file will be used.

                annotation_file:
                    Path to the csv file of the corresponding wave file.
                    If None, the predefined mini-test file will be used.

                silent:
                    Defines whether the sound should be played out loud or not.
                    Check the `use_input` and `use_output` parameter as well.

                use_input:
                    Specifies whether an input device is to be used or not.
                    Not necessary for evaluation mode if silent is True.

                use_output:
                    Specifies whether an output device is to be used or not.
                    Necessary for evaluation mode if silent is False.
            """

            if wav_file is None or annotation_file is None:
                DEFAULT_TEST = 'mini-test'
                wav_file = os.path.join(seds_constants.RES_TESTFILES_PATH, f'{DEFAULT_TEST}.wav')
                annotation_file = os.path.join(
                    seds_constants.RES_TESTFILES_PATH, f'{DEFAULT_TEST}.csv'
                )
            elif wav_file is None or annotation_file is None:
                raise FileNotFoundError('Paths to wave and corresponding csv file containing '
                                        'annotations have to be given correctly!')

            self.mode: SystemModes = SystemModes.EVALUATION
            self.wav_file = wav_file
            self.annotation_file = annotation_file
            self.silent = silent
            self.use_input = use_input
            self.use_output = use_output
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

    # noinspection PyPep8Naming
    # pylint:disable=invalid-name
    class conversion:
        """Cli for conversion actions.

        Actions:
            record_to_wav
        """

        # noinspection PyMethodMayBeStatic
        # pylint:disable=too-many-locals
        # pylint:disable=import-outside-toplevel
        # pylint:disable=no-self-use
        def record_to_wav(self,
                          path_storage_pickle: str,
                          target_wav_path: str,
                          sample_rate: int = 16000,
                          ):
            """Converts elements of a saved audio storage file (records-x.pickle) to a
            concatenated wave file.

            Args:
                path_storage_pickle:
                    Path to a recorded and saved records file. After running the system with using
                    `save_records=True` and `storage_length!=0`, a file containing the
                    recordings can be found in the resource folder.

                target_wav_path:
                    Specifies the target location and filename of the resulting wave file.

            Keyword Args:
                sample_rate:
                    Defines the used sample_rate.
            """
            import tensorflow as tf
            logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
            logging.root.setLevel(logging.INFO)
            _logger = logging.getLogger(__name__)

            max_bits_wav = 3.436e+10 - (12 + 24) * 8  # headers of wav
            max_float32_samples_wav = max_bits_wav // 32
            storage = file_utils.restore_audio_storage(path_storage_pickle)

            storage_elements = storage.get_elements()
            samples_of_element = len(storage_elements[0].received_samples)
            samples_in_buffer = len(storage_elements) * samples_of_element

            max_elements_in_wav = int(max_float32_samples_wav // samples_of_element)
            wav_files = audio_maths.round_up_div(samples_in_buffer, max_elements_in_wav)
            _logger.info(f'Creating {wav_files} wave files...')
            received_sample_lists = [element.received_samples for element in storage_elements]
            if wav_files > 1:
                wav_files_subs = [received_sample_lists[i:i + max_elements_in_wav]
                                  for i in
                                  range(0, len(received_sample_lists), max_elements_in_wav)
                                  ]
                for i, element in enumerate(wav_files_subs):
                    concat_samples = tf.concat(element, 0)
                    tensor_string = tf.audio.encode_wav(tf.expand_dims(concat_samples, -1),
                                                        sample_rate)
                    file_path = target_wav_path.split('.wav')[0] + f'_{i}.wav'
                    tf.io.write_file(file_path,
                                     tensor_string)
                    _logger.info(f'File {file_path} created.')
            else:
                concat_samples = tf.concat(received_sample_lists, 0)
                tensor_string = tf.audio.encode_wav(tf.expand_dims(concat_samples, -1),
                                                    sample_rate)
                tf.io.write_file(target_wav_path, tensor_string)
                _logger.info(f'File {target_wav_path} created.')
            _logger.info('Done')


def main():
    """Starts SedsCli script"""
    fire.Fire(SedsCli)


if __name__ == '__main__':
    main()
