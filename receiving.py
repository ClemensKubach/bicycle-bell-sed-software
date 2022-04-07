"""Module for receiving audio data."""

import csv
import queue
import threading
from abc import ABC, abstractmethod
from threading import Thread
import logging
import re
from typing import Optional, Protocol, Union, Type
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_io as tfio
import pyaudio
from configurations import ReceiverConfig


def _concat_samples(samples_chunk: list[tf.Tensor]) -> tf.Tensor:
    """Concatenate samples of the chunk"""
    return tf.concat(samples_chunk, 0)


@dataclass
class AudioReceiverElement(ABC):
    """AudioReceiverElement"""

    received_samples: tf.Tensor


@dataclass
class ProductionAudioReceiverElement(AudioReceiverElement):
    """ProductionAudioReceiverElement"""


@dataclass
class EvaluationAudioReceiverElement(AudioReceiverElement):
    """EvaluationAudioReceiverElement"""

    played_samples: tf.Tensor
    labels: tf.Tensor

    def __post_init__(self):
        if not (self.received_samples.shape == self.played_samples.shape and
                self.received_samples.shape == self.labels.shape):
            raise ValueError("Shapes of received_samples, played_samples and labels are not equal")


class AudioReceiverChunk(Protocol):
    """AudioReceiverChunk"""


class ProductionAudioReceiverChunk:
    """ProductionAudioReceiverChunk"""

    def __init__(self, elements_chunk: list[ProductionAudioReceiverElement]):
        self.elements_chunk = elements_chunk
        self.received_samples_chunk = _concat_samples(
            [element.received_samples for element in self.elements_chunk]
        )


class EvaluationAudioReceiverChunk:
    """EvaluationAudioReceiverChunk"""

    def __init__(self, elements_chunk: list[EvaluationAudioReceiverElement]):
        self.elements_chunk = elements_chunk
        self.received_samples_chunk = _concat_samples(
            [element.received_samples for element in self.elements_chunk]
        )
        self.played_samples_chunk = _concat_samples(
            [element.played_samples for element in self.elements_chunk]
        )
        self.labels_chunk = _concat_samples(
            [element.labels for element in self.elements_chunk]
        )


@dataclass
class AudioReceiverStorage:
    """AudioReceiverStorage

    Can result in a memory overflow. Use with caution.
    Set storage_size to 0 for no storage at all, and -1 for infinite storage."""

    storage_size: int

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self._storage = []
        self.keep_all = bool(self.storage_size < 0)

    def add_element(self, element: AudioReceiverElement):
        """Add element to the storage"""
        self._storage.append(element)
        if len(self._storage) > self.storage_size:
            self._storage.pop(0)

    def get_elements(self) -> list[AudioReceiverElement]:
        """Get elements from the storage"""
        return self._storage


@dataclass
class AudioReceiverBuffer:
    """AudioReceiverBuffer

    chunk_size: number of elements per chunk
    """

    cls: Type[Union[ProductionAudioReceiverChunk, EvaluationAudioReceiverChunk]]
    chunk_size: int

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self._buffer = queue.Queue[AudioReceiverElement]()

    @property
    def current_buffer_size(self) -> int:
        """Current buffer size"""
        return self._buffer.qsize()

    def add_element(self, element: AudioReceiverElement):
        """Add element to the buffer"""
        self._buffer.put(element)

    def get_latest_chunk(self) -> Optional[Union[ProductionAudioReceiverChunk,
                                                 EvaluationAudioReceiverChunk]]:
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


class AudioReceiver(Thread, ABC):
    """AudioReceiver"""

    def __init__(self, config: ReceiverConfig) -> None:
        super().__init__(daemon=True)
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.use_input = config.use_input
        self.input_device = config.input_device
        self.use_output = config.use_output
        self.output_device = config.output_device
        self.element_size = config.element_size
        self.chunk_size = config.chunk_size
        self.storage_size = config.storage_size

        self.frames_per_buffer = config.element_size
        self.window_size = config.chunk_size * config.element_size

        self.logger = logging.getLogger(__name__)
        self.storage = AudioReceiverStorage(self.storage_size)
        self.buffer = self._init_buffer()

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                rate=self.sample_rate,
                format=pyaudio.paFloat32,
                channels=self.channels,
                input=self.use_input,
                input_device_index=self.input_device,
                output=self.use_output,
                output_device_index=self.output_device,
                stream_callback=self._stream_callback,
                frames_per_buffer=self.frames_per_buffer,
                start=False,
            )
            self._stop_event = threading.Event()

            # show audio devices
            number_devices = self.pyaudio_instance.get_device_count()
            self.logger.info(f'Number of installed sound devices: {number_devices}')
            for i in range(number_devices):
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                self.logger.info(f'Sound device {i} info: {device_info}')
            input_device_info = self.pyaudio_instance.get_default_input_device_info()
            self.logger.info(f'Default input sound device info: {input_device_info}')
            output_device_info = self.pyaudio_instance.get_default_output_device_info()
            self.logger.info(f'Default output sound device info: {output_device_info}')

            # show selected devices
            self.logger.info(
                f'Selected input sound device index (None=default device): {config.input_device}')
            self.logger.info(
                f'Selected output sound device index (None=default device): {config.output_device}')

            self.logger.debug("AudioReceiver device initialized")
        except OSError:
            self.logger.error("Probability incompatible receiver "
                              "configuration for the selected device")
            raise
        except Exception:
            self.logger.error("An unknown error occurred")
            raise

    @abstractmethod
    def _init_buffer(self) -> AudioReceiverBuffer:
        """create buffer instance"""

    def run(self) -> None:
        """run the receiver in another thread"""
        self.stream.start_stream()
        self.logger.debug("AudioReceiver is running...")
        while self.stream.is_active() and not self._stop_event.is_set():
            pass

    def _stream_callback(self, in_data, frame_count, time_info, status):
        self.logger.debug(f"Audio data with {frame_count} samples "
                          f"for {time_info} and status {status} received")
        audio_as_np_float32 = np.fromstring(in_data, np.float32)[0::self.channels]
        # self._stream_callback_creator(audio_as_np_float32)
        element = AudioReceiverElement(tf.constant(audio_as_np_float32))
        self.buffer.add_element(element)
        return in_data, pyaudio.paContinue

    def receive_latest_chunk(self) -> Optional[AudioReceiverChunk]:
        """return latest chunk from buffer"""
        return self.buffer.get_latest_chunk()

    @property
    def delay(self) -> float:
        """delay in seconds"""
        return self.frames_per_buffer / self.sample_rate

    def close(self) -> None:
        """closes the receiver"""
        try:
            self._stop_event.set()
            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()
        except AttributeError:
            self.logger.debug("Not all streams could be closed, probability because of an "
                              "incomplete initialization of the AudioReceiver instance.")
            # no raise of an error necessary
        except Exception:
            self.logger.error("An unknown error occurred")
            raise
        finally:
            self.logger.debug("Audio receiver closed")


class ProductionAudioReceiver(AudioReceiver):
    """Production audio receiver"""

    def _init_buffer(self) -> AudioReceiverBuffer:
        return AudioReceiverBuffer(ProductionAudioReceiverChunk, self.chunk_size)


class EvaluationAudioReceiver(AudioReceiver):
    """Evaluation audio receiver"""

    def _init_buffer(self) -> AudioReceiverBuffer:
        return AudioReceiverBuffer(EvaluationAudioReceiverChunk, self.chunk_size)

    def __init__(self, config: ReceiverConfig,
                 wav_file: str, annotation_file: str, silent: bool) -> None:
        super().__init__(config)
        try:
            self.wav_file = wav_file
            self.annotation_file = annotation_file
            self.silent = silent

            self.wav = tf.cast(
                tf.squeeze(tfio.audio.AudioIOTensor(self.wav_file)[:, 0]), tf.float32
            ) / 32768.0
            self.annotations = []
            with open(annotation_file, 'r', newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                filename_wav = re.split(r'[/\\]', self.wav_file.split('.wav')[0])[-1]
                for row in csvreader:
                    if row[0] == filename_wav:
                        self.annotations = [
                            [float(s) for s in time_pair.split('#')]
                            for time_pair in row[1:]]
                        # tuple of start_time & end_time in sec for each pairs for each file
                        break
            sample_timings = []
            for seconds_timing_pair in self.annotations:
                start = int(seconds_timing_pair[0] * self.sample_rate)  # start time
                end = int(seconds_timing_pair[1] * self.sample_rate)  # end time
                sample_timings.append((start, end))
            self.sample_timings = np.zeros(shape=self.wav.shape)
            for start, end in tqdm(sample_timings):
                for i in range(start, end):
                    self.sample_timings[i] = True

            self.current_start_sample = 0
        except OSError:
            self.logger.error(f"File not found: {self.wav_file}")
            raise
        except Exception:
            self.logger.error("An unknown error occurred")
            raise

    def _stream_callback(self, in_data, frame_count, time_info, status):
        self.logger.debug(
            f"Audio data with {frame_count} samples for {time_info} and status {status} received")
        start_sample = self.current_start_sample
        end_sample = start_sample + self.frames_per_buffer
        try:
            played_samples = self.wav[start_sample:end_sample]
            if self.silent:
                received_samples = played_samples
            else:
                received_samples = np.fromstring(in_data, dtype=np.float32)[0::self.channels]
            labels = self.sample_timings[start_sample:end_sample]
            self.logger.debug(
                f"received Samples Shape {received_samples.shape} {labels.shape}")
            element = EvaluationAudioReceiverElement(
                tf.constant(received_samples),
                tf.constant(played_samples),
                tf.constant(labels)
            )
            self.buffer.add_element(element)
            self.current_start_sample += self.frames_per_buffer
            return played_samples, pyaudio.paContinue
        except IndexError:
            self.logger.debug("End of evaluation wave file reached")
            return [], pyaudio.paComplete
        except Exception:
            self.logger.error("An unknown error occurred")
            raise
