"""
Executable script for running the sound-event-detection system with different parametrizations.
"""

import logging
from typing import Union

from sed_system import utils
from sed_system.configurations import ModelSelection, SystemModes, LogLevels, SedSystemConfig, \
    ReceiverConfig, PredictorConfig, AudioConfig
from sed_system.inference_models import InferenceModels
from sed_system.predicting import ProductionPredictorResult, EvaluationPredictorResult
from sed_system.saved_models import SavedModels
from sed_system.sed_system import SedSystem


def main(tfmodel_path,
         mode: SystemModes = SystemModes.PRODUCTION,
         silent=False,
         use_input=True,
         use_output=False,
         storage_size=-1,
         wav_file=None,
         annotation_file=None,

         input_device=None,
         output_device=None,
         channels=1,
         gpu=False,
         save_records=True,

         sample_rate=16000,
         window_length=2.0,
         frame_length=0.02,

         loglevel: LogLevels = LogLevels.INFO,
         prob_logging=False):
    """Highly configurable execution script."""

    def custom_callback(
            predictor_result: Union[ProductionPredictorResult, EvaluationPredictorResult]):
        logger = logging.getLogger(__name__)
        window_time = utils.samples_to_seconds(
            sed.config.audio_config.window_size,
            sed.config.audio_config.sample_rate
        )

        if mode == SystemModes.PRODUCTION:
            res = predictor_result

            prob_print = f' [{res.result.probability:.2f}]' if prob_logging else ''
            logger.info(f'Prediction for the past {window_time}sec: '
                        f'{res.result.label}{prob_print} | delay: {res.result.delay.max_delay}sec '
                        f'with an inference time of {res.result.delay.inference}sec')
        else:
            res = predictor_result
            if prob_logging:
                received_print = f'{res.result.label} [{res.result.probability:.2f}]'
                played_print = f'{res.result_played.label} [{res.result_played.probability:.2f}]'
                gt_print = f'{res.result_ground_truth.label}' \
                           f' [{res.result_ground_truth.probability:.2f}]'
            else:
                received_print = f'{res.result.label}'
                played_print = f'{res.result_played.label}'
                gt_print = f'{res.result_ground_truth.label}'
            logger.info(f"Prediction of the past {window_time}sec: "
                        f"{received_print} received, {played_print} played, {gt_print} "
                        f"ground-truth | delay: {res.result.delay.max_delay}sec with an "
                        f"inference time of {res.result.delay.inference}sec")

    audio_config = AudioConfig(sample_rate, window_length, frame_length)

    system_config = SedSystemConfig(audio_config,
                                    mode,
                                    loglevel,
                                    gpu,
                                    save_records)
    sed = SedSystem(system_config)

    receiver_config = ReceiverConfig(audio_config,
                                     channels,
                                     use_input,
                                     input_device,
                                     use_output,
                                     output_device,
                                     storage_size)
    if mode == SystemModes.PRODUCTION:
        sed.system.init_receiver(receiver_config)
    elif mode == SystemModes.EVALUATION:
        sed.system.init_receiver(receiver_config, wav_file, annotation_file, silent)

    selected_model = ModelSelection(InferenceModels.TFLITE, SavedModels.CRNN)
    predictor_config = PredictorConfig(audio_config,
                                       tfmodel_path,
                                       selected_model,
                                       threshold=0.5,
                                       callback=custom_callback)
    sed.system.init_predictor(predictor_config)

    sed.start()


if __name__ == '__main__':
    # fire.Fire(main)
    main('models/crnn/')
