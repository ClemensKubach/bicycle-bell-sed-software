import logging
from typing import Union

from seds_cli.seds_lib.data.predictions.results import ProductionPredictorResult
from seds_cli.seds_lib.data.predictions.results import EvaluationPredictorResult


def get_custom_logging_callback(window_length, prob_logging):

    def custom_callback(
            predictor_result: Union[ProductionPredictorResult, EvaluationPredictorResult]):
        _logger = logging.getLogger(__name__)
        window_time = window_length
        res = predictor_result
        max_frame_delay = res.result.delay.predicting_delay.chunk_delay.max_in_buffer_waiting_time
        total_delay = res.result.delay.delay

        if isinstance(predictor_result, ProductionPredictorResult):
            prob_print = f' [{res.result.probability:.2f}]' if prob_logging else ''
            _logger.info(f'Prediction for the past {window_time:.3f}sec: '
                         f'{res.result.label}{prob_print} | delay: '
                         f'{total_delay - max_frame_delay:.3f}-{total_delay:.3f}sec')
        elif isinstance(predictor_result, EvaluationPredictorResult):
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
                         f'{total_delay - max_frame_delay:.3f}-{total_delay:.3f}sec')
        else:
            pass

    return custom_callback
