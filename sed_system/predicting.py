"""Predictor"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import Any

import tensorflow as tf

from sed_system import receiving
from sed_system.configurations import PredictorConfig


@dataclass(frozen=True)
class Delay:
    """Dataclass about the delay."""
    inference: float
    receiving: float

    @property
    def max_delay(self):
        """Sum of all delay parts."""
        return self.inference + self.receiving


@dataclass
class PredictorResult:
    """PredictorResult"""

    probability: float
    label: bool
    delay: Delay


@dataclass
class ProductionPredictorResult:
    """ProductionPredictorResult"""

    result: PredictorResult


@dataclass
class EvaluationPredictorResult:
    """EvaluationPredictorResult"""

    result: PredictorResult
    result_played: PredictorResult
    result_ground_truth: PredictorResult


class Predictor(Thread, ABC):
    """Predictor"""

    def __init__(self, config: PredictorConfig, initialized_receiver: receiving.AudioReceiver):
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
    def _predict(self, receiver_chunk: receiving.AudioReceiverChunk) -> Any:
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
            latest_chunk = self.receiver.receive_latest_chunk()
            if latest_chunk is not None:
                predictor_result = self._predict(latest_chunk)
                self._run_callback(predictor_result)

    def close(self):
        """close predictor"""
        self._stop_event.set()


class ProductionPredictor(Predictor):
    """ProductionPredictor"""

    def _predict(self, receiver_chunk: receiving.ProductionAudioReceiverChunk
                 ) -> ProductionPredictorResult:
        samples = receiver_chunk.received_samples_chunk
        predictor_result_received = self._predict_for_samples(samples)
        return ProductionPredictorResult(predictor_result_received)


class EvaluationPredictor(Predictor):
    """EvaluationPredictor"""

    def _predict(self, receiver_chunk: receiving.EvaluationAudioReceiverChunk
                 ) -> EvaluationPredictorResult:
        predictor_result_received = self._predict_for_samples(receiver_chunk.received_samples_chunk)
        predictor_result_played = self._predict_for_samples(receiver_chunk.played_samples_chunk)

        gt_prob = receiver_chunk.labels_chunk
        gt_label = bool(gt_prob)
        predictor_result_gt = PredictorResult(gt_prob, gt_label, Delay(.0, self.receiver.delay))

        return EvaluationPredictorResult(predictor_result_received,
                                         predictor_result_played,
                                         predictor_result_gt)
