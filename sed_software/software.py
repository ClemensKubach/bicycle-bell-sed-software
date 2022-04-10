"""Module containing system classes."""

import logging
import os
import threading
import time
from typing import Union
import keyboard

from sed_software.data.configs.configs import SedSoftwareConfig
from sed_software.systems import ProductionSedSystem, EvaluationSedSystem
from sed_software.utils import save_receiver_storage
from sed_software.selectors.selectors import SystemModes, LogLevels


class SedSoftware:
    """System for sound event detection"""

    def __init__(self, software_config: SedSoftwareConfig):
        self.config = software_config
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
            logger_format = '%(asctime)s %(levelname)s: %(message)s'
            self._separate_delay_log = False
        elif loglevel == LogLevels.DEBUG:
            logger.setLevel(logging.DEBUG)
            console.setLevel(logging.DEBUG)
            logger_format = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
            self._separate_delay_log = True
        elif loglevel == LogLevels.ERROR:
            logger.setLevel(logging.ERROR)
            console.setLevel(logging.ERROR)
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
                    save_receiver_storage(self.system.receiver.storage, './records/')
