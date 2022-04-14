"""Module containing file util functions."""

import pickle
import time

from seds_lib.storage.audio.persistent import AudioStorage


def save_audio_storage(storage: AudioStorage, path: str = '') -> None:
    """save storage in file"""
    timestamp = time.strftime('%Y.%m.%d-%H.%M')
    with open(f'{path}receiverBuffer-{timestamp}.pickle', 'wb') as file:
        pickle.dump(storage, file)


def restore_audio_storage(filepath: str) -> AudioStorage:
    """restore storage from file"""
    with open(filepath, 'rb') as file:
        storage = pickle.load(file)
    return storage
