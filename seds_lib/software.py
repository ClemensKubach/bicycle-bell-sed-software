"""Module containing system classes."""

import logging
import os
import threading
import time
from typing import Union

from seds_lib.data.configs.configs import SedSoftwareConfig
from seds_lib.systems import ProductionSedSystem, EvaluationSedSystem
from seds_lib.utils import file_utils
from seds_lib.selectors.selectors import SystemModes, LogLevels


class SedSoftware:
    """Top level software component for sound event detection."""

    def __init__(self, software_config: SedSoftwareConfig):
        self.config = software_config
        audio = self.config.audio_config

        self._setup_logger(self.config.loglevel)
        self._setup_system(self.config.system_mode, audio.sample_rate, audio.chunk_size)
        self._show_gpu_setting(self.config.gpu)

    def _show_gpu_setting(self, gpu):
        """Procedure for enabling or disabling visibility of the gpu devices for tensorflow."""
        env_var = "CUDA_VISIBLE_DEVICES"
        if gpu:
            os.environ[env_var] = "0"
            self._logger.info(f'Environment variable {env_var} is set to 0')
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self._logger.info(f'Environment variable {env_var} is set to -1')

    def _setup_system(self, system_mode: SystemModes, sample_rate: int, chunk_size: int):
        """Procedure for initializing the system depending on the chosen system_mode."""
        self.system_mode = system_mode
        self.system: Union[ProductionSedSystem, EvaluationSedSystem]
        if system_mode == SystemModes.PRODUCTION:
            self.system = ProductionSedSystem(sample_rate, chunk_size)
        elif system_mode == SystemModes.EVALUATION:
            self.system = EvaluationSedSystem(sample_rate, chunk_size)
        else:
            msg = f'Mode {system_mode} is not a valid mode! It must be of type SystemModes.'
            self._logger.error(msg)
            raise ValueError(msg)

    def _setup_logger(self, loglevel: LogLevels):
        """Procedure for initializing the logging config."""
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
        self._logger = logging.getLogger(__name__)

    def start(self):
        """Starts all system components.
        Runs on the main thread without a lot of computational load."""
        if self.system.receiver is None or self.system.predictor is None:
            msg = 'Receiver and Predictor have to be initialized before running!'
            self._logger.error(msg)
            raise UnboundLocalError(msg)

        self._logger.debug("Worker started")
        self._logger.info('Press Ctrl+C or Interrupt the Kernel')
        self.system.receiver.start()
        self.system.predictor.start()

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._logger.warning("Caught KeyboardInterrupt")
            self.system.receiver.close()
            self.system.predictor.close()
            self._logger.info('Stopped gracefully')
            pickle_lock = threading.Lock()
            with pickle_lock:
                if self.config.save_records and self.system.receiver.storage.current_size > 0:
                    file_utils.save_audio_storage(self.system.receiver.storage, './records/')
