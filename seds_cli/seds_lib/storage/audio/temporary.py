"""Module for a temporary audio storage"""

import logging
import threading
from typing import Type, Union, Optional

from seds_cli.seds_lib.data.audio.chunks import ProductionAudioChunk
from seds_cli.seds_lib.data.audio.chunks import EvaluationAudioChunk
from seds_cli.seds_lib.data.audio.elements import AudioElement


class AudioBuffer:
    """Buffer for audio elements. It is used as resource between the Receiver and Predictor Threads.

    Thus, thread-safety is provided.

    Args:
        cls
            class name of ProductionAudioChunk or EvaluationAudioChunk
        chunk_size
            number of elements per chunk
    """

    def __init__(self,
                 cls: Type[Union[ProductionAudioChunk, EvaluationAudioChunk]],
                 chunk_size: int):
        self.cls = cls
        self.chunk_size = chunk_size

        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._buffer: list[AudioElement] = []
        self._previous_slice: Optional[list[AudioElement]] = None

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
        with self._lock:
            self._buffer = []

    def _get_latest_n_slice(self, n: int):
        """get the last n elements of the buffer as list slice"""
        with self._lock:
            return self._buffer[-n:]

    def get_latest_chunk(self) -> Optional[Union[ProductionAudioChunk,
                                                 EvaluationAudioChunk]]:
        """Get the latest chunk from the buffer with concatenated samples.

        Notes:
            TODO Some elements can get lost
        """
        currently_buffered = self.current_buffer_size
        if self._previous_slice is None and currently_buffered < self.chunk_size:
            return None
        if currently_buffered == 0:
            return None
        extracting_count = min(currently_buffered, self.chunk_size)
        # if buffer is larger than chunk_size, just take the newest for not increasing the delay
        # negative effect: some samples can get lost, but only if the predictor delay is
        # larger as the ...
        latest_slice = self._get_latest_n_slice(extracting_count)
        self.reset()
        complement_size = self.chunk_size - extracting_count
        complement_slice = [] if complement_size == 0 else self._previous_slice[-complement_size:]
        new_slice: list = complement_slice + latest_slice
        self._previous_slice = new_slice
        # TODO lost elements proving and note
        return self.cls(new_slice, extracting_count)
