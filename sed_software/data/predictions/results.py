"""Prediction results"""

from dataclasses import dataclass

from sed_software.data.time.delay import Delay


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
