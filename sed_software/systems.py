"""Module containing system classes."""

import logging
from abc import ABC
from typing import Optional

from sed_software.data.configs.configs import ReceiverConfig
from sed_software.workers.predicting import ProductionPredictor, EvaluationPredictor, \
    PredictorConfig
from sed_software.workers.receiving import ProductionAudioReceiver, EvaluationAudioReceiver


class ModeSedSystem(ABC):
    """ModeSedSystem"""

    def __init__(self, sample_rate: int, chunk_size: int):
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

        self.logger = logging.getLogger(__name__)


class ProductionSedSystem(ModeSedSystem):
    """ProductionSedSystem"""

    def __init__(self, sample_rate: int, chunk_size: int):
        super().__init__(sample_rate, chunk_size)

        self.logger.info('Production mode selected')
        self.predictor: Optional[ProductionPredictor] = None
        self.receiver: Optional[ProductionAudioReceiver] = None

    def init_receiver(self, config: ReceiverConfig) -> None:
        """init production receiver"""
        self.receiver = ProductionAudioReceiver(config)

    def init_predictor(self, config: PredictorConfig) -> None:
        """init production predictor"""
        self.predictor = ProductionPredictor(config, self.receiver)


class EvaluationSedSystem(ModeSedSystem):
    """EvaluationSedSystem"""

    def __init__(self, sample_rate: int, chunk_size: int):
        super().__init__(sample_rate, chunk_size)

        self.logger.info('Evaluation mode selected')
        self.predictor: Optional[EvaluationPredictor] = None
        self.receiver: Optional[EvaluationAudioReceiver] = None

    def init_receiver(self, config: ReceiverConfig,
                      wav_file: str,
                      annotation_file: str,
                      silent: bool) -> None:
        """init evaluation receiver"""
        silent_txt = 'with' if silent else 'without'
        self.logger.info(
            f'Evaluation mode selected {silent_txt} using the direct input of the wave file'
        )
        self.receiver = EvaluationAudioReceiver(config, wav_file, annotation_file, silent)

    def init_predictor(self, config: PredictorConfig) -> None:
        """init evaluation predictor"""
        self.predictor = EvaluationPredictor(config, self.receiver)
