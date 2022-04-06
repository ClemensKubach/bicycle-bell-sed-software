"""
Executable script for running the sound-event-detection system with different parametrizations.
"""

import logging
from typing import Union

import fire

from predicting import ProductionPredictorResult, EvaluationPredictorResult, PredictorConfig
from receiving import ReceiverConfig
from sed_system import SedSystem, LogLevels, SystemModes, SedSystemConfig


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
         sample_rate=22050,

         chunk_size=20,

         loglevel: LogLevels = LogLevels.INFO,
         prob_logging=False):
    """Highly configurable execution script."""

    element_size = int(sample_rate * 0.1)

    def custom_callback(
            predictor_result: Union[ProductionPredictorResult, EvaluationPredictorResult]):
        logger = logging.getLogger(__name__)

        if mode == SystemModes.PRODUCTION:
            res = predictor_result
            prob_print = f' [{res.result.probability:.2f}]' if prob_logging else ''
            logger.info(f'Prediction for the past {sed.system.receiver.window_size}sec: '
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
            logger.info(f"Prediction of the past {sed.system.receiver.window_size}sec: "
                        f"{received_print} received, {played_print} played, {gt_print} "
                        f"ground-truth | delay: {res.result.delay.max_delay}sec with an "
                        f"inference time of {res.result.delay.inference}sec")

    system_config = SedSystemConfig(mode,
                                    loglevel,
                                    gpu,
                                    save_records,
                                    sample_rate,
                                    chunk_size)
    sed = SedSystem(system_config)

    receiver_config = ReceiverConfig(sample_rate,
                                     channels,
                                     use_input,
                                     input_device,
                                     use_output,
                                     output_device,
                                     element_size,
                                     chunk_size,
                                     storage_size)
    if mode == SystemModes.PRODUCTION:
        sed.system.init_receiver(receiver_config)
    elif mode == SystemModes.EVALUATION:
        sed.system.init_receiver(receiver_config, wav_file, annotation_file, silent)

    predictor_config = PredictorConfig(tfmodel_path,
                                       sample_rate,
                                       chunk_size,
                                       threshold=0.5,
                                       callback=custom_callback)
    sed.system.init_predictor(predictor_config)

    sed.start()


if __name__ == '__main__':
    #fire.Fire(main)
    main('models/crnn/')
