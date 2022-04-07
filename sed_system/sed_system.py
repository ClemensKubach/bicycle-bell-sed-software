"""Module containing system classes."""

import logging
import os
import threading
import time
from abc import ABC
from typing import Optional, Union
import keyboard

import utils
from lib.configurations import SedSystemConfig, SystemModes, LogLevels
from lib.predicting import ProductionPredictor, EvaluationPredictor, PredictorConfig
from lib.receiving import ProductionAudioReceiver, EvaluationAudioReceiver, ReceiverConfig


class SedSystem:
    """System for sound event detection"""

    def __init__(self, system_config: SedSystemConfig):
        self.config = system_config
        audio = self.config.audio_config

        self._setup_logger(self.config.loglevel)
        self._setup_system(self.config.system_mode, audio.sample_rate, audio.chunk_size)
        self._show_gpu_setting(self.config.gpu)
        self._stop_event = threading.Event()

    def _show_gpu_setting(self, gpu):
        env_var = "CUDA_VISIBLE_DEVICES"
        if gpu:
            os.environ[env_var] = "0"
            self.logger.info(f'Environment variable {env_var} is set to 0')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.logger.info(f'Environment variable {env_var} is set to -1')

    def _setup_system(self, system_mode: SystemModes, sample_rate: int, chunk_size: int):
        self.system_mode = system_mode
        self.system: Union[ProductionSedSystem, EvaluationSedSystem]
        if system_mode == SystemModes.PRODUCTION:
            self.system = ProductionSedSystem(sample_rate, chunk_size)
        elif system_mode == SystemModes.EVALUATION:
            self.system = EvaluationSedSystem(sample_rate, chunk_size)
        else:
            msg = f'Mode {system_mode} is not a valid mode! It must be of type SystemModes.'
            self.logger.error(msg)
            raise ValueError(msg)

    def _setup_logger(self, loglevel: LogLevels):
        logger = logging.getLogger()
        console = logging.StreamHandler()
        if loglevel == LogLevels.INFO:
            logger.setLevel(logging.INFO)
            console.setLevel(logging.INFO)
            logger_format = '%(asctime)s %(levelname)s %(module)s: %(message)s'
            self._separate_delay_log = False
        elif loglevel == LogLevels.DEBUG:
            logger.setLevel(logging.DEBUG)
            console.setLevel(logging.DEBUG)
            logger_format = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
            self._separate_delay_log = True
        else:
            msg = f'Loglevel {loglevel} is not a valid loglevel! It must be of type LogLevels.'
            raise ValueError(msg)

        timestamp = time.strftime('%Y.%m.%d-%H.%M.%S')
        logging.basicConfig(format=logger_format, filename=f'./logs/{timestamp}.log')
        formatter = logging.Formatter(logger_format)
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        self.logger = logging.getLogger(__name__)

    def start(self):
        """Starts the system"""
        if self.system.receiver is None or self.system.predictor is None:
            msg = 'Receiver and Predictor have to be initialized before running!'
            self.logger.error(msg)
            raise UnboundLocalError(msg)

        self.logger.debug("Worker started")
        self.logger.info('Press Ctrl+C or Interrupt the Kernel')
        self.system.receiver.start()
        self.system.predictor.start()

        try:
            while not self._stop_event.is_set():
                if keyboard.is_pressed('q'):
                    self._stop_event.set()
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            self.logger.warning("Caught KeyboardInterrupt")
            self.logger.info('Stopped gracefully')
        finally:
            pickle_lock = threading.Lock()
            with pickle_lock:
                self.system.receiver.close()
                self.system.predictor.close()
                if self.config.save_records:
                    utils.save_receiver_storage(self.system.receiver.storage, './records/')


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
