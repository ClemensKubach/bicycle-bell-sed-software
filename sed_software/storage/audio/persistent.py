"""persistent storage"""

import logging

from sed_software.data.audio.elements import AudioElement


class AudioStorage:
    """AudioStorage

    Can result in a memory overflow. Use with caution.
    Set storage_size to 0 for no storage at all, and -1 for infinite storage."""

    def __init__(self, storage_size: int):
        self.storage_size = storage_size

        self.logger = logging.getLogger(__name__)
        self._storage = []
        self.keep_all = bool(self.storage_size < 0)

    def add_element(self, element: AudioElement):
        """Add element to the storage"""
        self._storage.append(element)
        if len(self._storage) > self.storage_size:
            self._storage.pop(0)

    def get_elements(self) -> list[AudioElement]:
        """Get elements from the storage"""
        return self._storage
