"""Predictor"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any

import tensorflow as tf

from sed_software.data.configs.configs import PredictorConfig
from sed_software.data.predictions.results import PredictorResult, ProductionPredictorResult, \
    EvaluationPredictorResult
from sed_software.data.time.delay import Delay
from sed_software.workers.receiving import AudioReceiver, AudioChunk, \
    ProductionAudioChunk, EvaluationAudioChunk


class Predictor(Thread, ABC):
    """Predictor"""

    def __init__(self, config: PredictorConfig, initialized_receiver: AudioReceiver):
        super().__init__(daemon=True)
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.receiver = initialized_receiver

        if self.receiver is None:
            msg = 'Receiver must be initialized before initializing Predictor!'
            self.logger.error(msg)
            raise UnboundLocalError(msg)

        model_selection = self.config.model_selection
        self.model = model_selection.inference_model(
            model_selection.saved_model(self.config.tfmodel_path, self.config.threshold),
            self.receiver.config.audio_config.window_size
        )
        self._stop_event = threading.Event()
        self.logger.debug("Predictor initialized")

    def _predict_for_samples(self, samples: tf.Tensor) -> PredictorResult:
        time_start = time.perf_counter()
        audio = self.config.audio_config
        y_pred_prob, y_pred_label = self.model.inference(samples, audio.sample_rate)
        time_end = time.perf_counter()
        return PredictorResult(y_pred_prob,
                               y_pred_label,
                               Delay(time_end - time_start, self.receiver.delay))

    @abstractmethod
    def _predict(self, receiver_chunk: AudioChunk) -> Any:
        """Predict"""

    def _run_callback(self, predictor_result: Any):
        """runs callback"""
        if self.config.callback is None:
            pass
        else:
            self.config.callback(predictor_result)

    def run(self):
        """run the predictor in another thread"""
        while not self._stop_event.is_set():
            #a = time.perf_counter()
            latest_chunk = self.receiver.receive_latest_chunk()
            #b = time.perf_counter()
            if latest_chunk is not None:
                #print('R', b - a)
                predictor_result = self._predict(latest_chunk)
                self._run_callback(predictor_result)
                # TODO stop Callback Time in Log

    def close(self):
        """close predictor"""
        self._stop_event.set()


class ProductionPredictor(Predictor):
    """ProductionPredictor"""

    def _predict(self, receiver_chunk: ProductionAudioChunk
                 ) -> ProductionPredictorResult:
        samples = receiver_chunk.received_samples_chunk
        predictor_result_received = self._predict_for_samples(samples)
        return ProductionPredictorResult(predictor_result_received)


class EvaluationPredictor(Predictor):
    """EvaluationPredictor"""

    def _predict(self, receiver_chunk: EvaluationAudioChunk
                 ) -> EvaluationPredictorResult:
        predictor_result_received = self._predict_for_samples(receiver_chunk.received_samples_chunk)
        predictor_result_played = self._predict_for_samples(receiver_chunk.played_samples_chunk)

        gt_prob = receiver_chunk.labels_chunk
        gt_label = bool(gt_prob)
        predictor_result_gt = PredictorResult(gt_prob, gt_label, Delay(.0, self.receiver.delay))

        return EvaluationPredictorResult(predictor_result_received,
                                         predictor_result_played,
                                         predictor_result_gt)
