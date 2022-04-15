"""Module for predictors"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any, Union

import tensorflow as tf

from seds_cli.seds_lib.data.audio.chunks import AudioChunk
from seds_cli.seds_lib.data.audio.chunks import ProductionAudioChunk
from seds_cli.seds_lib.data.audio.chunks import EvaluationAudioChunk
from seds_cli.seds_lib.data.configs.configs import PredictorConfig
from seds_cli.seds_lib.data.predictions.results import PredictorResult
from seds_cli.seds_lib.data.predictions.results import ProductionPredictorResult
from seds_cli.seds_lib.data.predictions.results import EvaluationPredictorResult
from seds_cli.seds_lib.data.time.delay import Delay
from seds_cli.seds_lib.data.time.delay import PredictorDelay
from seds_cli.seds_lib.data.time.delay import ChunkDelay

from seds_cli.seds_lib.workers.receiving import AudioReceiver


class Predictor(Thread, ABC):
    """Abstract base Predictor as Thread"""

    def __init__(self, config: PredictorConfig, initialized_receiver: AudioReceiver):
        """Initialize Predictor Thread with a PredictorConfig and a prepared AudioReceiver instance.
        """
        super().__init__(daemon=False, name='Predictor-Thread')
        self.config = config
        self._logger = logging.getLogger(__name__)

        self.receiver = initialized_receiver

        if self.receiver is None:
            msg = 'Receiver must be initialized before initializing Predictor!'
            self._logger.error(msg)
            raise UnboundLocalError(msg)

        model_selection = self.config.model_selection
        self.model = model_selection.inference_model(
            model_selection.saved_model(self.config.tfmodel_path, self.config.threshold),
            self.receiver.config.audio_config.window_size
        )
        self._stop_event = threading.Event()
        self._logger.debug("Predictor initialized")

    def _predict_for_samples(self, samples: tf.Tensor,
                             chunk_delay: ChunkDelay) -> PredictorResult:
        """Apply inference on samples tensor and create PredictorResult instance."""
        time_start = time.perf_counter()
        audio = self.config.audio_config
        y_pred_prob, y_pred_label = self.model.inference(samples, audio.sample_rate)
        time_end = time.perf_counter()
        inference_time = time_end - time_start
        predictor_delay = PredictorDelay(chunk_delay, inference_time)
        return PredictorResult(y_pred_prob,
                               y_pred_label,
                               Delay(self.receiver.delay, predictor_delay))

    @abstractmethod
    def _predict(self, audio_chunk: AudioChunk, chunk_delay: ChunkDelay) -> Any:
        """Defines how the sample tensors from the given AudioChunk are extracted.
        _predict_for_samples is applied on the tensors and
        the resulting PredictorResult is getting wrapped.
        """

    @abstractmethod
    def _default_callback(self, predictor_result: Union[ProductionPredictorResult,
                                                        EvaluationPredictorResult]):
        """If no callback function is specified in the config, this will be applied by default."""

    def _run_callback(self, predictor_result: Any):
        """Decides whether a custom callback or the default callback function is getting applied."""
        if self.config.callback is None:
            # default callback
            self._default_callback(predictor_result)
        else:
            # custom callback
            self.config.callback(predictor_result)

    def run(self):
        """Defines what the predictor thread has to do while running."""
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            latest_chunk = self.receiver.receive_latest_chunk()
            end_time = time.perf_counter()
            if latest_chunk is not None:
                processing_time = end_time - start_time
                max_frame_delay = latest_chunk.num_unseen * self.config.audio_config.frame_length
                chunk_delay = ChunkDelay(processing_time, max_frame_delay)
                predictor_result = self._predict(latest_chunk, chunk_delay)
                self._run_callback(predictor_result)

    def close(self):
        """Stops the predictor"""
        self._stop_event.set()
        self.join(timeout=5)


class ProductionPredictor(Predictor):
    """Production mode of the Predictor."""

    def _default_callback(self, predictor_result: Union[ProductionPredictorResult,
                                                        EvaluationPredictorResult]):
        res = predictor_result
        window_time = self.config.audio_config.window_length
        max_frame_delay = res.result.delay.predicting_delay.chunk_delay.max_in_buffer_waiting_time
        total_delay = res.result.delay.delay

        prob_print = f' [{res.result.probability:.2f}]'
        self._logger.info(f'Prediction for the past {window_time:.3f}sec: '
                          f'{res.result.label}{prob_print} | delay: '
                          f'{total_delay - max_frame_delay:.3f}-{total_delay:.3f}sec')

    def _predict(self, audio_chunk: ProductionAudioChunk,
                 chunk_delay: ChunkDelay) -> ProductionPredictorResult:
        samples = audio_chunk.received_samples_chunk
        predictor_result_received = self._predict_for_samples(samples, chunk_delay)
        return ProductionPredictorResult(predictor_result_received)


class EvaluationPredictor(Predictor):
    """Evaluation mode of the Predictor.

    Makes predictions not only for the received data but also for the direct
    data from the wave file without playing it.

    The from the receiver passed ground-truth data is handled as well.
    """

    def _default_callback(self, predictor_result: Union[ProductionPredictorResult,
                                                        EvaluationPredictorResult]):
        res = predictor_result
        window_time = self.config.audio_config.window_length
        max_frame_delay = res.result.delay.predicting_delay.chunk_delay.max_in_buffer_waiting_time
        total_delay = res.result.delay.delay

        received_print = f'{res.result.label} [{res.result.probability:.2f}]'
        played_print = f'{res.result_played.label} [{res.result_played.probability:.2f}]'
        gt_print = f'{res.result_ground_truth.label}' \
                   f' [{res.result_ground_truth.probability:.2f}]'
        self._logger.info(f'Prediction of the past {window_time:.3f}sec: '
                          f'{received_print} received, {played_print} played, {gt_print} '
                          f'ground-truth | delay: '
                          f'{total_delay - max_frame_delay:.3f}-{total_delay:.3f}sec')

    def _predict(self, audio_chunk: EvaluationAudioChunk,
                 chunk_delay: ChunkDelay) -> EvaluationPredictorResult:
        predictor_result_received = self._predict_for_samples(audio_chunk.received_samples_chunk,
                                                              chunk_delay)
        predictor_result_played = self._predict_for_samples(audio_chunk.played_samples_chunk,
                                                            chunk_delay)

        gt_prob = audio_chunk.labels_chunk
        gt_label = bool(gt_prob)
        gt_delay = Delay(self.receiver.delay, PredictorDelay(chunk_delay, 0))
        predictor_result_gt = PredictorResult(gt_prob, gt_label, gt_delay)

        return EvaluationPredictorResult(predictor_result_received,
                                         predictor_result_played,
                                         predictor_result_gt)
