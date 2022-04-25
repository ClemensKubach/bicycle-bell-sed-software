"""Module for a temporary audio storage"""

import logging
import threading
from typing import Type, Union, Optional, List

from seds_cli.seds_lib.data.audio.chunks import ProductionAudioChunk
from seds_cli.seds_lib.data.audio.chunks import EvaluationAudioChunk
from seds_cli.seds_lib.data.audio.elements import AudioElement
from seds_cli.seds_lib.data.configs.configs import AudioConfig


class AudioBuffer:
    """Buffer for audio elements. It is used as resource between the Receiver and Predictor Threads.

    Thus, thread-safety is provided.

    Args:
        cls
            class name of ProductionAudioChunk or EvaluationAudioChunk
        audio_config
            audio config data.
    """

    def __init__(self,
                 cls: Type[Union[ProductionAudioChunk, EvaluationAudioChunk]],
                 audio_config: AudioConfig):
        self.cls = cls
        self.audio_config = audio_config

        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._buffer: List[AudioElement] = []
        self._previous_slice: Optional[List[AudioElement]] = None

    @property
    def current_buffer_size(self) -> int:
        """Current buffer size"""
        with self._lock:
            size = len(self._buffer)
        return size

    def add_element(self, element: AudioElement):
        """Add element to the buffer"""
        with self._lock:
            self._buffer.append(element)

    def reset(self):
        """clear buffer"""
        self._buffer = []

    def _get_latest_n_slice(self, num: int):
        """get the last n elements of the buffer as list slice"""
        return self._buffer[-num:]

    def extract_latest_chunk(self) -> Optional[Union[ProductionAudioChunk,
                                                     EvaluationAudioChunk]]:
        """Get the latest chunk from the buffer with concatenated samples.

        Notes:
            If the delay is greater than window_length, the samples older than
            window_length seconds will be ignored. This prevents a permanent effect of an
            unexpected and spontaneous lag on the delay of the system.
        """
        with self._lock:
            currently_buffered = len(self._buffer)
            if self._previous_slice is None and currently_buffered < self.audio_config.chunk_size:
                return None
            if currently_buffered == 0:
                return None
            if currently_buffered > self.audio_config.chunk_size:
                lost_seconds = (currently_buffered - self.audio_config.chunk_size) * \
                               self.audio_config.frame_length
                self._logger.warning(f'The buffer was full and {lost_seconds:.3f} seconds were '
                                     f'skipped from {self.audio_config.window_length:.3f} seconds '
                                     'ago to avoid increasing the delay permanently.')
            extracting_count = min(currently_buffered, self.audio_config.chunk_size)
            latest_slice = self._get_latest_n_slice(extracting_count)
            self.reset()
        complement_size = self.audio_config.chunk_size - extracting_count
        complement_slice = [] if complement_size == 0 else self._previous_slice[-complement_size:]
        new_slice: list = complement_slice + latest_slice
        self._previous_slice = new_slice
        return self.cls(new_slice, extracting_count)
