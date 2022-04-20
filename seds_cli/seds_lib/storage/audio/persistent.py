"""Module for a persistent audio storage"""

import logging
from typing import Union, List

from seds_cli.seds_lib.utils import audio_maths
from seds_cli.seds_lib.data.audio.elements import ProductionAudioElement
from seds_cli.seds_lib.data.audio.elements import EvaluationAudioElement
from seds_cli.seds_lib.data.audio.elements import AudioElement
from seds_cli.seds_lib.data.configs.configs import AudioConfig


class AudioStorage:
    """Persistent audio storage for storing

    Currently, the individual audio elements can be collected in this storage in memory while
    running the program. At stopping the system, the data is stored persistently on the local drive.

    Set storage_size to 0 for no storage at all, and -1 for infinite storage.
    Can result in a memory overflow at runtime. Use with caution.
    """

    def __init__(self, storage_length: float, audio_config: AudioConfig):
        if storage_length <= 0:
            self.storage_size = storage_length
        else:
            num_samples = audio_maths.seconds_to_samples(storage_length, audio_config.sample_rate)
            self.storage_size = int(num_samples / audio_config.frame_size)

        self._logger = logging.getLogger(__name__)
        self._storage = []
        self.keep_all = bool(self.storage_size < 0)

    @property
    def current_size(self):
        return len(self._storage)

    def add_element(self, element: Union[AudioElement,
                                         ProductionAudioElement,
                                         EvaluationAudioElement]):
        """Add element to the storage"""
        if self.keep_all or len(self._storage) < self.storage_size:
            self._storage.append(element)

    def get_elements(self) -> List[Union[ProductionAudioElement, EvaluationAudioElement]]:
        """Get list of elements from the storage"""
        return self._storage
