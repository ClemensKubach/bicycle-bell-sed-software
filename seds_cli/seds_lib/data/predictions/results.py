"""Module with all data classes about results of the predictor."""

from dataclasses import dataclass

from seds_cli.seds_lib.data.time.delay import Delay


@dataclass
class PredictorResult:
    """Data class containing the computation results of the predictor.

    Attributes

    probability (float)
        return value of the model. Typically, between 0 and 1.
    label (bool)
        label as True or False depending on how the probability value relates to the set threshold.
    delay (Delay)
        contains information about composition of the various time delays.
    """

    probability: float
    label: bool
    delay: Delay


@dataclass
class ProductionPredictorResult:
    """Data class containing the computation results of the predictor in production mode.

    Attributes

    result (PredictorResult)
        wrapped prediction result for received data.
    """

    result: PredictorResult


@dataclass
class EvaluationPredictorResult:
    """Data class containing the computation results of the predictor in evaluation mode.

    Attributes

    result (PredictorResult)
        wrapped prediction result for received data.
    result_played (PredictorResult)
        wrapped prediction result for played data
        (wave data is directly accessed without using microphone and speaker).
    result_ground_truth (PredictorResult)
        wrapped ground-truth data.
    """

    result: PredictorResult
    result_played: PredictorResult
    result_ground_truth: PredictorResult
