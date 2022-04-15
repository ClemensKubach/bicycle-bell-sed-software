"""
Executable script for running the sound-event-detection system with different parametrizations.
"""

import logging
from typing import Union

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
from seds_cli.seds_lib.data.predictions.results import ProductionPredictorResult
from seds_cli.seds_lib.data.predictions.results import EvaluationPredictorResult


def main(tfmodel_path,
         mode: SystemModes = SystemModes.PRODUCTION,
         silent=False,
         use_input=True,
         use_output=False,
         storage_length=0,
         wav_file=None,
         annotation_file=None,

         input_device=None,
         output_device=None,
         channels=1,
         gpu=False,
         save_records=False,

         sample_rate=16000,
         window_length=2.0,
         frame_length=0.001,

         threshold=0.5,
         callback=None,
         infer_model: InferenceModels = InferenceModels.TFLITE,
         saved_model: SavedModels = SavedModels.CRNN,

         loglevel: LogLevels = LogLevels.INFO,
         prob_logging=False):
    """Highly configurable execution script."""

    def custom_callback(
            predictor_result: Union[ProductionPredictorResult, EvaluationPredictorResult]):
        _logger = logging.getLogger(__name__)
        window_time = sed.config.audio_config.window_length
        res = predictor_result
        max_frame_delay = res.result.delay.predicting_delay.chunk_delay.max_in_buffer_waiting_time
        total_delay = res.result.delay.delay

        if mode == SystemModes.PRODUCTION:
            prob_print = f' [{res.result.probability:.2f}]' if prob_logging else ''
            _logger.info(f'Prediction for the past {window_time:.3f}sec: '
                         f'{res.result.label}{prob_print} | delay: '
                         f'{total_delay-max_frame_delay:.3f}-{total_delay:.3f}sec')
        else:
            if prob_logging:
                received_print = f'{res.result.label} [{res.result.probability:.2f}]'
                played_print = f'{res.result_played.label} [{res.result_played.probability:.2f}]'
                gt_print = f'{res.result_ground_truth.label}' \
                           f' [{res.result_ground_truth.probability:.2f}]'
            else:
                received_print = f'{res.result.label}'
                played_print = f'{res.result_played.label}'
                gt_print = f'{res.result_ground_truth.label}'
            _logger.info(f'Prediction of the past {window_time:.3f}sec: '
                         f'{received_print} received, {played_print} played, {gt_print} '
                         f'ground-truth | delay: '
                         f'{total_delay-max_frame_delay:.3f}-{total_delay:.3f}sec')

    audio_config = AudioConfig(sample_rate, window_length, frame_length)

    system_config = SedSoftwareConfig(audio_config,
                                      mode,
                                      loglevel,
                                      gpu,
                                      save_records)
    sed = SedSoftware(system_config)

    receiver_config = ReceiverConfig(audio_config,
                                     channels,
                                     use_input,
                                     input_device,
                                     use_output,
                                     output_device,
                                     storage_length)
    if mode == SystemModes.PRODUCTION:
        sed.system.init_receiver(receiver_config)
    elif mode == SystemModes.EVALUATION:
        sed.system.init_receiver(receiver_config, wav_file, annotation_file, silent)

    selected_model = ModelSelection(infer_model, saved_model)
    predictor_config = PredictorConfig(audio_config,
                                       tfmodel_path,
                                       selected_model,
                                       threshold,
                                       callback)
    sed.system.init_predictor(predictor_config)

    sed.start()


if __name__ == '__main__':
    fire.Fire(main)
