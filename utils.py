import pickle
import time
import numpy as np
import librosa

#from receiver import AudioReceiverBuffer

def get_frameLength(frame_size, sample_rate):
    """ Returns windowLength=frameLength=n_fft in SampleTiming format"""
    return int(np.ceil(frame_size*sample_rate)) # 46 ms frame in sample count

def get_hopLength(hop_size, sample_rate):
    return int(np.ceil(hop_size*sample_rate)) # 23 ms overlap in sample count

def convert_frameChunkSize_elementChunkSize(chunk_size: int, frame_length: int, hop_length: int):
    a = librosa.frames_to_samples(chunk_size, hop_length=hop_length, n_fft=frame_length)
    b = int(a / frame_length)
    return b

def add_flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    return result

def saveReceiverBuffer(receiverBuffer, path: str='') -> None:
    timestamp = time.strftime(f'%Y.%m.%d-%H.%M')
    with open(path+f"receiverBuffer-{timestamp}.pickle", "wb") as f:
        pickle.dump(receiverBuffer, f)

# def savePredicterResultBuffer(predicterBuffer: AudioReceiverBuffer, path: str='') -> None:
#     timestamp = time.localtime()
#     with open(path+f"predicterBuffer-{timestamp}.pickle", "wb") as f:
#         pickle.dump(predicterBuffer, f)

def chunkTime(hop_size: float, chunk_size: int):
    return hop_size * chunk_size

def roundUpDiv(f, d):
    return f // d + (f % d > 0)

def restoreReceiverBuffer(filepath: str):
    with open(filepath, 'rb') as f:
        receiverBuffer = pickle.load(f)
    return receiverBuffer

