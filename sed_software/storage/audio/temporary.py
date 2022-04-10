"""temporary storage"""

import logging
import queue
from typing import Type, Union, Optional

from sed_software.data.audio.chunks import ProductionAudioChunk, EvaluationAudioChunk
from sed_software.data.audio.elements import AudioElement


class AudioBuffer:
    """AudioBuffer

    chunk_size: number of elements per chunk
    """

    def __init__(self,
                 cls: Type[Union[ProductionAudioChunk, EvaluationAudioChunk]],
                 chunk_size: int):
        self.cls = cls
        self.chunk_size = chunk_size

        self.logger = logging.getLogger(__name__)
        self._buffer = queue.Queue[AudioElement]()

    @property
    def current_buffer_size(self) -> int:
        """Current buffer size"""
        return self._buffer.qsize()

    def add_element(self, element: AudioElement):
        """Add element to the buffer"""
        self._buffer.put(element)

    def get_latest_chunk(self) -> Optional[Union[ProductionAudioChunk,
                                                 EvaluationAudioChunk]]:
        """Get the latest chunk from the buffer with concatenated samples."""
        if self.current_buffer_size < self.chunk_size:
            self.logger.warning(
                f"Not enough frames in buffer for chunk creation. "
                f"{self.chunk_size} needed, but only {self.current_buffer_size} given.")
            return None
        latest_slice = []
        for _ in range(self.chunk_size):
            latest_slice.insert(0, self._buffer.get())
        return self.cls(latest_slice)
