"""Executable Script
"""

import fire

from bicycle_bell_seds import callbacks
from seds_cli.seds_cli import SedsCli
from seds_cli.seds_lib.selectors.selectors import InferenceModels
from seds_cli.seds_lib.selectors.selectors import SavedModels
from seds_cli.seds_lib.selectors.selectors import LogLevels


class DesktopBicycleBellSedsCli(SedsCli):
    """CLI of the Sound Event Detection Software
    designed for signals of a bicycle bell as target sound, running on a desktop machine.

    Usage:
        Instead of specifying the path to a predefined saved_model in tfmodel_path,
        it can be accessed directly using !model_name (like !crnn).
        The value for saved_model will be inferred automatically, if not specified separately.

        If running this script, a tensorflow lite inference model will be chosen for infer_model
        and the gpu is not used.
        Because the selected model should be one of {CRNN, YAMNET_BASE, YAMNET_EXTENDED} as
        selected in saved_model, a sample_rate of 16 kHz is predefined.
        Default channel number is set to 1 for a mono mic.

    Functionality:
        There are 2 modes to run the system. The **Production** and the **Evaluation** mode.

        With the **Production** mode, the sounds of the environment are recorded and evaluated with
        the help of the selected microphone.

        In the **Evaluation** mode, the system can be tested using a selected wave file and
        an associated CSV file, containing start and end time of the contiguous presence of the
        target sound event in each line. The output contains an indication of the annotated
        ground-truth value from the CSV,
        a prediction value for the played wave data and a prediction value for the recorded audio.
        If `silent=True`, it is no audio played out loud.

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
            Default at 2 for a stereo mic.

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

        input_device:
            Index of the device using for input. None defines the system default device.
            Default is None.

        output_device:
            Index of the device using for output. None defines the system default device.
            Default is None.

        window_length:
            Value in seconds. A larger value means more context information which can be helpful
            for the recognition by the model.
            Typically, as long as the sound event takes at a maximum.

        frame_length:
            Value in seconds, smaller than window_length. Smaller Values can decrease the delay
            between receiving the sound and getting a prediction result for it.
            A too small value can result in bad performance or a ValueError.

        loglevel:
            Defines at which logging level, the logging commands should be perceived.
            Currently, supported is one of {DEBUG, INFO, ERROR}.
            Default is INFO.

        save_log:
            Defines whether the log output should be stored on disk. Default is False.
    """

    # pylint:disable=too-many-arguments
    # pylint:disable=too-many-locals
    # pylint:disable=duplicate-code
    def __init__(self,
                 tfmodel_path: str,

                 threshold: float = 0.5,
                 channels: int = 2,

                 gpu: bool = False,
                 infer_model: InferenceModels = InferenceModels.TFLITE,
                 saved_model: SavedModels = SavedModels.CRNN,

                 storage_length: int = 0,
                 save_records: bool = False,

                 input_device: int = None,
                 output_device: int = None,

                 window_length: float = 1.0,
                 frame_length: float = 0.001,

                 loglevel: LogLevels = LogLevels.INFO,
                 save_log: bool = False,

                 prob_logging: bool = False,
                 ):
        sample_rate = 16000
        callback = callbacks.get_custom_logging_callback(window_length, prob_logging)

        super().__init__(
            tfmodel_path,
            threshold,
            channels,
            gpu,
            infer_model,
            saved_model,
            storage_length,
            save_records,
            input_device,
            output_device,
            sample_rate,
            window_length,
            frame_length,
            callback,
            loglevel,
            save_log,
        )


def main():
    """Starts optimized SedsCli script designed for default desktop use and
    detection of bicycle bell sound events."""
    fire.Fire(DesktopBicycleBellSedsCli)


if __name__ == '__main__':
    main()
