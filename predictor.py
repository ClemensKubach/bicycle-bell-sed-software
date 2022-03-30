"""Predictor"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import tensorflow as tf

import receiver
from inference_models import TFLiteInferenceModel
from saved_models import CrnnSavedModel


@dataclass
class PredictorResult:
    """PredictorResult"""

    probability: float
    label: bool
    delay: float

    def __post_init__(self):
        self.delay = round(self.delay*1000)/1000


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


@dataclass
class Predictor(ABC):
    """Predictor"""

    tfmodel_path: str
    sample_rate: int
    chunk_size: int
    threshold: float = 0.5

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = TFLiteInferenceModel(CrnnSavedModel(self.tfmodel_path, self.threshold))
        self.logger.debug("Predictor initialized")

    def _predict_for_samples(self, samples: tf.Tensor) -> PredictorResult:
        time_start = time.perf_counter()
        y_pred_prob, y_pred_label = self.model.inference(samples, self.sample_rate)
        time_end = time.perf_counter()
        return PredictorResult(y_pred_prob, y_pred_label, time_end - time_start)

    @abstractmethod
    def predict(self, receiver_chunk: receiver.AudioReceiverChunk) -> PredictorResult:
        """Predict"""


@dataclass
class ProductionPredictor(Predictor):
    """ProductionPredictor"""

    def predict(self,
                receiver_chunk: receiver.ProductionAudioReceiverChunk) -> ProductionPredictorResult:
        samples = receiver_chunk.receivedSamplesChunk
        predictor_result_received = self._predict_for_samples(samples)
        return ProductionPredictorResult(predictor_result_received)


@dataclass
class EvaluationPredictor(Predictor):
    """EvaluationPredictor"""

    def predict(self,
                receiver_chunk: receiver.EvaluationAudioReceiverChunk) -> EvaluationPredictorResult:
        predictor_result_received = self._predict_for_samples(receiver_chunk.receivedSamplesChunk)
        predictor_result_played = self._predict_for_samples(receiver_chunk.playedSamplesChunk)

        gt_prob = receiver_chunk.labelsChunk
        gt_label = bool(gt_prob)
        predictor_result_gt = PredictorResult(gt_prob, gt_label, 0.0)

        return EvaluationPredictorResult(predictor_result_received,
                                         predictor_result_played,
                                         predictor_result_gt)
