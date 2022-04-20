"""Executable Script
"""

import os

import fire

from bicycle_bell_seds import callbacks
from seds_cli import seds_constants
from seds_cli.seds_cli import SedsCli
from seds_cli.seds_lib.selectors.selectors import InferenceModels
from seds_cli.seds_lib.selectors.selectors import SavedModels
from seds_cli.seds_lib.selectors.selectors import LogLevels


class DesktopBicycleBellSedsCli(SedsCli):

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

                 window_length: float = 2.0,
                 frame_length: float = 0.001,

                 loglevel: LogLevels = LogLevels.INFO,

                 prob_logging: bool = False,
                 ):
        if tfmodel_path in [f'!{model.name.lower()}' for model in list(SavedModels)]:
            selection = tfmodel_path.split('!')[1]
            tfmodel_path = os.path.join(seds_constants.RES_MODELS_PATH, selection)
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
            use_input,
            use_output,
            input_device,
            output_device,
            sample_rate,
            window_length,
            frame_length,
            callback,
            loglevel
        )


def main():
    fire.Fire(DesktopBicycleBellSedsCli)


if __name__ == '__main__':
    main()
