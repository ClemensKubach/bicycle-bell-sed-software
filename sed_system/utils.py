"""Utils"""

import pickle
import time
from typing import Union

from sed_system.receiving import AudioReceiverStorage


def samples_to_seconds(samples: Union[int, list], sample_rate: int):
    """return time in seconds of sample size or samples list"""
    if isinstance(samples, list):
        length = len(samples)
    else:
        length = samples
    return length / sample_rate


def round_up_div(num_a, num_b):
    """division with round-up(num_a // num_b)"""
    return num_a // num_b + (num_a % num_b > 0)


def save_receiver_storage(storage: AudioReceiverStorage, path: str = '') -> None:
    """save storage in file"""
    timestamp = time.strftime('%Y.%m.%d-%H.%M')
    with open(f'{path}receiverBuffer-{timestamp}.pickle', 'wb') as file:
        pickle.dump(storage, file)


def restore_receiver_storage(filepath: str):
    """restore storage from file"""
    with open(filepath, 'rb') as file:
        storage = pickle.load(file)
    return storage
