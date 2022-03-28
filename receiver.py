import csv
from threading import Thread
import time
import librosa
import logging
import pyaudio
import numpy as np
import re
from typing import Optional, Union
from tqdm import tqdm
import utils


class AudioReceiverElement(object):

    def __init__(self, receivedSamples: np.ndarray) -> None:
        self.receivedSamples = receivedSamples

class ProductionAudioReceiverElement(AudioReceiverElement):

    def __init__(self, receivedSamples: np.ndarray) -> None:
        super().__init__(receivedSamples)

class EvaluationAudioReceiverElement(AudioReceiverElement):

    def __init__(self, receivedSamples: np.ndarray, playedSamples: np.ndarray, labelsOfSamples: np.ndarray) -> None:
        super().__init__(receivedSamples)
        assert receivedSamples.shape == playedSamples.shape == labelsOfSamples.shape
        self.playedSamples = playedSamples
        self.labelsOfSample = labelsOfSamples





class AudioReceiverChunk(object):

    def __init__(self, chunkOfElements: list[AudioReceiverElement]) -> None:
        self.chunkOfElements = chunkOfElements
        self.isUsable = False

    def concatenateElements(self) -> None:
        self.receivedSamplesChunk = np.concatenate([element.receivedSamples for element in self.chunkOfElements])

class ProductionAudioReceiverChunk(AudioReceiverChunk):

    def __init__(self, chunkOfElements: list[ProductionAudioReceiverElement]) -> None:
        super().__init__(chunkOfElements)

class EvaluationAudioReceiverChunk(AudioReceiverChunk):

    def __init__(self, chunkOfElements: list[EvaluationAudioReceiverElement]) -> None:
        super().__init__(chunkOfElements)

    def concatenateElements(self) -> None:
        super().concatenateElements()
        self.playedSamplesChunk = np.concatenate([element.playedSamples for element in self.chunkOfElements])
        self.labelsChunk = np.concatenate([element.labelsOfSample for element in self.chunkOfElements])




class AudioReceiverBuffer(object):

    def __init__(self, bufferMaxSize: int, chunk_size: int, frame_length: int, hop_length: int) -> None:
        self.logger = logging.getLogger(__name__)
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.elementChunkSize = utils.convert_frameChunkSize_elementChunkSize(chunk_size, frame_length, hop_length) 
        self.logger.debug(f"Number of elements for equal number of samples to a chunk size of {chunk_size}: {self.elementChunkSize}")
        if bufferMaxSize < 0:
            self.keepAll = True
        else:
            self.keepAll = False
        try:
            assert self.keepAll or bufferMaxSize >= self.elementChunkSize
        except AssertionError:
            self.logger.warning("The bufferMaxSize has to be as large as the elementChunkSize at a minimum. bufferMaxSize is set to the elementChunkSize automatically.")
            bufferMaxSize = self.elementChunkSize
        self.bufferMaxSize = bufferMaxSize
        self._buffer = []
        self._chunkBuffer = []
        self._chunkBufferUnusable = []
        self._chunkBufferUnprocessed = []
        self.thread = Thread(target=self._prepareUnusableChunks, daemon=True)

        self.thread.start()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['thread'], state['logger']
        return state

    def __setstate__(self, state):
        # Restore instance attribute
        self.__dict__.update(state)
        self.thread = Thread(target=self._prepareUnusableChunks, daemon=True)
        self.logger = logging.getLogger(__name__)

    def _prepareUnusableChunks(self):
        while True:
            if len(self._chunkBufferUnusable) > 0:
                c: AudioReceiverChunk = self._chunkBufferUnusable.pop(0)
                if not c.isUsable:
                    c.concatenateElements()
                    c.isUsable = True
                    self._chunkBufferUnprocessed.append(c)
            time.sleep(0.00001)

    def getBufferSize(self) -> int:
        return len(self._buffer)

    def addAudioReceiverElement(self, element: AudioReceiverElement) -> None:
        if not self.keepAll and len(self._buffer) > self.bufferMaxSize:
            self._buffer.pop(0)
        self._buffer.append(element)
        if len(self._buffer) >= self.elementChunkSize:
            chunk = AudioReceiverChunk(self._buffer[-self.elementChunkSize:])
            self._chunkBuffer.append(chunk)
            self._chunkBufferUnusable.append(chunk)

    def getChunkFromIndex(self, frameIndex: int) -> Union[AudioReceiverChunk, None]:
        try:
            assert frameIndex+self.elementChunkSize <= self.getBufferSize()
            result = AudioReceiverChunk(self._buffer[frameIndex:frameIndex+self.elementChunkSize])
        except AssertionError:
            self.logger.warning(f"Not enough frames in buffer for chunk creation. {self.elementChunkSize} needed, but only {self.getBufferSize()} given.")
            result = None
        finally:
            return result

    def getLastChunk(self) -> Union[AudioReceiverChunk, None]:
        try:
            assert self.elementChunkSize <= self.getBufferSize()
            result = AudioReceiverChunk(self._buffer[-self.elementChunkSize:])
        except AssertionError:
            self.logger.warning(f"Not enough frames in buffer for chunk creation. {self.elementChunkSize} needed, but only {self.getBufferSize()} given.")
            result = None
        finally:
            return result

    def getUsableChunksNumber(self):
        return len(self._chunkBufferUnprocessed)

    def getUnusableChunksNumber(self):
        return len(self._chunkBufferUnusable)

    def getNextUsableChunk(self) -> Union[AudioReceiverChunk, None]:
        if self.getUsableChunksNumber() > 0 and self._chunkBufferUnprocessed[0].isUsable:
            return self._chunkBufferUnprocessed.pop(0)
        else: return None

class ProductionAudioReceiverBuffer(AudioReceiverBuffer):

    def __init__(self, bufferMaxSize: int, chunk_size: int, frame_length: int, hop_length: int) -> None:
        super().__init__(bufferMaxSize, chunk_size, frame_length, hop_length)

    def addAudioReceiverElement(self, element: ProductionAudioReceiverElement) -> None:
        if not self.keepAll and len(self._buffer) > self.bufferMaxSize:
            self._buffer.pop(0)
        self._buffer.append(element)
        if len(self._buffer) >= self.elementChunkSize:
            l = self._buffer[-self.elementChunkSize:]
            chunk = ProductionAudioReceiverChunk(l)
            self._chunkBuffer.append(chunk)
            self._chunkBufferUnusable.append(chunk)

    def getChunkFromIndex(self, frameIndex: int) -> Union[ProductionAudioReceiverChunk, None]:
        try:
            assert frameIndex+self.elementChunkSize <= self.getBufferSize()
            result = ProductionAudioReceiverChunk(self._buffer[frameIndex:frameIndex+self.elementChunkSize])
        except AssertionError:
            self.logger.warning(f"Not enough frames in buffer for chunk creation. {self.elementChunkSize} needed, but only {self.getBufferSize()} given.")
            result = None
        return result

    def getLastChunk(self) -> Union[ProductionAudioReceiverChunk, None]:
        try:
            assert self.elementChunkSize <= self.getBufferSize()
            result = ProductionAudioReceiverChunk(self._buffer[-self.elementChunkSize:])
        except AssertionError:
            self.logger.warning(f"Not enough frames in buffer for chunk creation. {self.elementChunkSize} needed, but only {self.getBufferSize()} given.")
            result = None
        return result
    
    def getNextUsableChunk(self) -> Union[ProductionAudioReceiverChunk, None]:
        return super().getNextUsableChunk()

class EvaluationAudioReceiverBuffer(AudioReceiverBuffer):

    def __init__(self, bufferMaxSize: int, chunk_size: int, frame_length: int, hop_length: int) -> None:
        super().__init__(bufferMaxSize, chunk_size, frame_length, hop_length)

    def addAudioReceiverElement(self, element: EvaluationAudioReceiverElement) -> None:
        if not self.keepAll and len(self._buffer) > self.bufferMaxSize:
            self._buffer.pop(0)
        self._buffer.append(element)
        if len(self._buffer) >= self.elementChunkSize:
            chunk = EvaluationAudioReceiverChunk(self._buffer[-self.elementChunkSize:])
            self._chunkBuffer.append(chunk)
            self._chunkBufferUnusable.append(chunk)

    def getChunkFromIndex(self, frameIndex: int) -> Union[EvaluationAudioReceiverChunk, None]:
        try:
            assert frameIndex+self.elementChunkSize <= self.getBufferSize()
            result = EvaluationAudioReceiverChunk(self._buffer[frameIndex:frameIndex+self.elementChunkSize])
        except AssertionError:
            self.logger.warning(f"Not enough frames in buffer for chunk creation. {self.elementChunkSize} needed, but only {self.getBufferSize()} given.")
            result = None
        return result

    def getLastChunk(self) -> Union[EvaluationAudioReceiverChunk, None]:
        try:
            assert self.elementChunkSize <= self.getBufferSize()
            result = EvaluationAudioReceiverChunk(self._buffer[-self.elementChunkSize:])
        except AssertionError:
            self.logger.warning(f"Not enough frames in buffer for chunk creation. {self.elementChunkSize} needed, but only {self.getBufferSize()} given.")
            result = None
        return result

    def getNextUsableChunk(self) -> Union[EvaluationAudioReceiverChunk, None]:
        return super().getNextUsableChunk()






class AudioReceiver(object):
    
    def __init__(self, buffer: AudioReceiverBuffer, sample_rate: int, frame_length: int, hop_length: int, channels: int) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing AudioReceiver")
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.channels = channels
        self.buffer = buffer
        self.frames_per_bufferElement = self.frame_length*self.channels
        self.logger.debug("AudioReceiver initialized")

    def _stream_callback(self, in_data, frame_count, time_info, status):
        self.logger.debug(f"Audio data with {frame_count} samples for {time_info} and status {status} received")
        # extract channel 0 data from 2 channels, if you want to extract channel 1, please change to [1::2]
        audio_as_np_float32 = np.fromstring(in_data,dtype=np.float32)[0::self.channels]
        self._stream_callback_creator(audio_as_np_float32)
        return (in_data, pyaudio.paContinue)

    def _stream_callback_creator(self, receivedSamples: np.ndarray) -> None:
        element = AudioReceiverElement(receivedSamples)
        self.buffer.addAudioReceiverElement(element)

    def initReceiver(self, input: bool, output: bool, inputDevice_index: Optional[int]=None, outputDevice_index: Optional[int]=None) -> None:
        try:
            self.pyaudioInstance = pyaudio.PyAudio()
            self.stream = self.pyaudioInstance.open(
                rate=self.sample_rate,
                format=pyaudio.paFloat32,
                channels=self.channels,
                input=input,
                input_device_index=inputDevice_index,
                output=output,
                output_device_index=outputDevice_index,
                stream_callback=self._stream_callback,
                frames_per_buffer=self.frames_per_bufferElement
            )

            # show audio devices
            numberOfDevices = self.pyaudioInstance.get_device_count()
            self.logger.info(f'Number of installed sound devices: {numberOfDevices}')
            for i in range(numberOfDevices):
                deviceInfo = self.pyaudioInstance.get_device_info_by_index(i)
                self.logger.info(f'Sound device {i} info: {deviceInfo}')
            inputDeviceInfo = self.pyaudioInstance.get_default_input_device_info()
            self.logger.info(f'Default input sound device info: {inputDeviceInfo}')
            outputDeviceInfo = self.pyaudioInstance.get_default_output_device_info()
            self.logger.info(f'Default output sound device info: {outputDeviceInfo}')

            # show selected devices
            self.logger.info(f'Selected input sound device index (None=default device): {inputDevice_index}')
            self.logger.info(f'Selected output sound device index (None=default device): {outputDevice_index}')

            self.stream.start_stream()
            self.logger.debug("AudioReceiver device initialized")
        except OSError:
            self.logger.error(f"Probabiliy incompatible receiver configuration for the selected device")
            raise 
        except Exception:
            self.logger.error(f"An unknown error occured")
            raise

    def receiveNextChunkOfBuffer(self) -> Union[AudioReceiverChunk, None]:
        return self.buffer.getNextUsableChunk()

    def getDelay(self, seperate=False):
        a = self.buffer.getUsableChunksNumber()
        b = self.buffer.getUnusableChunksNumber()
        timeFactor = round(self.frames_per_bufferElement / self.sample_rate * 1000) / 1000
        if seperate:
            return {'usable': a*timeFactor, 'unusable': b*timeFactor}
        else:
            return (a + b) * timeFactor

    def closeReceiver(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.pyaudioInstance.terminate()
        except AttributeError:
            self.logger.debug(f"Not all streams could be closed, probabiliy because of an incomplete initialization of the AudioReceiver instance")
            # no raise of an error necessary
        except Exception:
            self.logger.error(f"An unknown error occured")
            raise
        finally:
            self.logger.debug("Audio receiver closed")

class ProductionAudioReceiver(AudioReceiver):

    def __init__(self, buffer: ProductionAudioReceiverBuffer, sample_rate: int, frame_length: int, hop_length: int, channels: int) -> None:
        super().__init__(buffer, sample_rate, frame_length, hop_length, channels)
        self.buffer = buffer

    def _stream_callback_creator(self, receivedSamples: np.ndarray) -> None:
        element = ProductionAudioReceiverElement(receivedSamples)
        self.buffer.addAudioReceiverElement(element)

    def receiveNextChunkOfBuffer(self) -> Union[ProductionAudioReceiverChunk, None]:
        return super().receiveNextChunkOfBuffer()

class EvaluationAudioReceiver(AudioReceiver):

    def __init__(self, buffer: EvaluationAudioReceiverBuffer, sample_rate: int, frame_length: int, hop_length: int, channels: int, wavFilepath: str, annotationFilepath: str, silent: bool) -> None:
        super().__init__(buffer, sample_rate, frame_length, hop_length, channels)
        self.buffer = buffer

        try:
            self.wavFilepath = wavFilepath
            self.annotationFilepath = annotationFilepath
            self.silent = silent

            self.wav = librosa.load(wavFilepath, sr=sample_rate, dtype=np.float32)[0]
            self.annotations = []
            with open(annotationFilepath, 'r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                filenameWav =  re.split(r'/|\\', wavFilepath.split('.wav')[0])[-1]
                for row in csvreader:
                    if row[0] == filenameWav:
                        self.annotations = [[float(s) for s in time_pair.split('#')] for time_pair in row[1:]] # tuple of start_time & end_time in sec for each pairs for each file
                        break
            sampleTimings = []
            for secondsTimingPair in self.annotations:
                s = librosa.time_to_samples(secondsTimingPair[0], sr=sample_rate) # start time
                e = librosa.time_to_samples(secondsTimingPair[1], sr=sample_rate) # end time
                sampleTimings.append((s, e))
            self.sampleTimings = np.zeros(shape=self.wav.shape)
            for s, e in tqdm(sampleTimings):
                for i in range(s, e, 1):
                    self.sampleTimings[i] = True

            self.currentStartSample = 0
        except OSError:
            self.logger.error(f"File not found: {self.wavFilepath}")
            raise
        except Exception:
            self.logger.error(f"An unknown error occured")
            raise

    def _stream_callback(self, in_data, frame_count, time_info, status):
        self.logger.debug(f"Audio data with {frame_count} samples for {time_info} and status {status} received")
        startSample = self.currentStartSample
        endSample = startSample + self.frames_per_bufferElement
        try:
            playedSamples = self.wav[startSample:endSample]
            if self.silent: 
                receivedSamples = playedSamples
            else: 
                receivedSamples = np.fromstring(in_data,dtype=np.float32)[0::self.channels]
            labelsOfSamples = self.sampleTimings[startSample:endSample]
            self.logger.debug(f"received Samples Shape {receivedSamples.shape} {labelsOfSamples.shape}")
            self._stream_callback_creator(receivedSamples, playedSamples, labelsOfSamples)
        except IndexError:
            self.logger.debug(f"End of evaluation wave file reached")
        except Exception:
            self.logger.error(f"An unknown error occured")
            raise
        finally:
            self.currentStartSample += self.frames_per_bufferElement
            return playedSamples, pyaudio.paContinue

    def _stream_callback_creator(self, receivedSamples: np.ndarray, playedSamples: np.ndarray, labelsOfSamples: np.ndarray) -> None:
        element = EvaluationAudioReceiverElement(receivedSamples, playedSamples, labelsOfSamples)
        self.buffer.addAudioReceiverElement(element)

    def receiveNextChunkOfBuffer(self) -> Union[EvaluationAudioReceiverChunk, None]:
        return super().receiveNextChunkOfBuffer()
    
