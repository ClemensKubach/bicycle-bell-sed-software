"""Module for receivers."""

import csv
import threading
import time
from abc import ABC, abstractmethod
from threading import Thread
import logging
import re
from typing import Optional, Union
from random import randint

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_io as tfio
import pyaudio

from seds_cli.seds_lib.data.audio.chunks import ProductionAudioChunk
from seds_cli.seds_lib.data.audio.chunks import EvaluationAudioChunk
from seds_cli.seds_lib.data.audio.elements import AudioElement
from seds_cli.seds_lib.data.audio.elements import EvaluationAudioElement
from seds_cli.seds_lib.data.configs.configs import ReceiverConfig
from seds_cli.seds_lib.data.time.delay import ReceiverDelay
from seds_cli.seds_lib.storage.audio.persistent import AudioStorage
from seds_cli.seds_lib.storage.audio.temporary import AudioBuffer


class AudioReceiver(Thread, ABC):
    """Abstract base AudioReceiver as Thread.
    Expects a ReceiverConfig instance.

    It initializes an AudioStorage and AudioBuffer instance. Then PortAudio via PyAudio is used
    for opening a stream such that callbacks for each frame can be called asynchronous.

    References:
        https://people.csail.mit.edu/hubert/pyaudio/
    """

    def __init__(self, config: ReceiverConfig) -> None:
        super().__init__(daemon=False, name='Receiver-Thread')
        self.config = config

        audio = self.config.audio_config

        self._logger = logging.getLogger(__name__)
        self.storage = AudioStorage(
            self.config.storage_length,
            self.config.audio_config
        )
        self.buffer = self._init_buffer()

        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            self._stream = self._pyaudio_instance.open(
                rate=audio.sample_rate,
                format=pyaudio.paFloat32,
                channels=self.config.channels,
                input=self.config.use_input,
                input_device_index=self.config.input_device,
                output=self.config.use_output,
                output_device_index=self.config.output_device,
                stream_callback=self._stream_callback,
                frames_per_buffer=audio.frame_size,
                start=False,
            )
            self._lock = threading.Lock()
            self._stop_event = threading.Event()
            self._stream_callback_time: float = -1.0
            self._measure_time: bool = True

            # show audio devices
            number_devices = self._pyaudio_instance.get_device_count()
            self._logger.info(f'Number of installed sound devices: {number_devices}')
            for i in range(number_devices):
                device_info = self._pyaudio_instance.get_device_info_by_index(i)
                self._logger.info(f'Sound device {i} info: {device_info}')
            input_device_info = self._pyaudio_instance.get_default_input_device_info()
            self._logger.info(f'Default input sound device info: {input_device_info}')
            output_device_info = self._pyaudio_instance.get_default_output_device_info()
            self._logger.info(f'Default output sound device info: {output_device_info}')

            # show selected devices
            self._logger.info(
                f'Selected input sound device index (None=default device): {config.input_device}')
            self._logger.info(
                f'Selected output sound device index (None=default device): {config.output_device}')

            self._logger.debug("AudioReceiver device initialized")
        except OSError:
            self._logger.error("Probability incompatible receiver "
                               "configuration for the selected device")
            raise
        except Exception:
            self._logger.error("An unknown error occurred")
            raise

    @abstractmethod
    def _init_buffer(self) -> AudioBuffer:
        """Creates a buffer instance."""

    def run(self) -> None:
        """Defines what the receiver thread has to do while running.

        A time measurement is initiated at random intervals to more accurately
        determine the system delay.
        """
        self._stream.start_stream()
        self._logger.debug("AudioReceiver is running...")
        while not self._stop_event.is_set() and self._stream.is_active():
            with self._lock:
                self._measure_time = True
            time.sleep(randint(1, 5))

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """Defines the pyAudio callback function.
        This is called each time, a new frame of size frame_length (frame_size) is received.
        Then it is transformed into a new AudioElement instance.

        Time needed for this should be smaller than the length of a frame (frame_length).
        For this, the time is measured for random samples and saved into the delay attribute
        of the receiver (access via receiver_instance.delay).
        """
        start_time = None
        with self._lock:
            if self._measure_time:
                start_time = time.perf_counter()
        audio_as_np_float32 = np.fromstring(in_data, np.float32)[0::self.config.channels]
        element = AudioElement(tf.constant(audio_as_np_float32))
        self.buffer.add_element(element)
        self.storage.add_element(element)
        with self._lock:
            if self._measure_time and start_time is not None:
                end_time = time.perf_counter()
                self._stream_callback_time = end_time - start_time
                self._measure_time = False
        return in_data, pyaudio.paContinue

    def receive_latest_chunk(self) -> Optional[Union[ProductionAudioChunk,
                                                     EvaluationAudioChunk]]:
        """Returns the latest chunk from the buffer. Should be called from another thread."""
        chunk = self.buffer.get_latest_chunk()
        return chunk

    @property
    def delay(self) -> ReceiverDelay:
        """Delay information"""
        with self._lock:
            return ReceiverDelay(self._stream_callback_time)

    def close(self) -> None:
        """Stops the receiver."""
        try:
            self._stop_event.set()
            self._stream.stop_stream()
            self._stream.close()
            self._pyaudio_instance.terminate()
            self.join(timeout=5)
        except AttributeError:
            self._logger.debug("Not all streams could be closed, probability because of an "
                               "incomplete initialization of the AudioReceiver instance.")
            # no raise of an error necessary
        except Exception:
            self._logger.error("An unknown error occurred")
            raise
        finally:
            self._logger.debug("Audio receiver closed")


class ProductionAudioReceiver(AudioReceiver):
    """Production mode AudioReceiver as Thread.
    Expects a ReceiverConfig instance.

    It initializes an AudioStorage and AudioBuffer instance. Then PortAudio via PyAudio is used
    for opening a stream such that callbacks for each frame can be called asynchronous.

    References:
        https://people.csail.mit.edu/hubert/pyaudio/
    """

    def _init_buffer(self) -> AudioBuffer:
        audio = self.config.audio_config
        return AudioBuffer(ProductionAudioChunk, audio.chunk_size)


class EvaluationAudioReceiver(AudioReceiver):
    """Evaluation mode AudioReceiver as Thread.
    Expects a ReceiverConfig instance.

    Loads the wav_file and the corresponding annotation_file.

    It initializes an AudioStorage and AudioBuffer instance. Then PortAudio via PyAudio is used
    for opening a stream such that callbacks for each frame can be called asynchronous.

    References:
        https://people.csail.mit.edu/hubert/pyaudio/
    """

    def _init_buffer(self) -> AudioBuffer:
        audio = self.config.audio_config
        return AudioBuffer(EvaluationAudioChunk, audio.chunk_size)

    def __init__(self, config: ReceiverConfig,
                 wav_file: str, annotation_file: str, silent: bool) -> None:
        super().__init__(config)
        audio = self.config.audio_config
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
                start = int(seconds_timing_pair[0] * audio.sample_rate)  # start time
                end = int(seconds_timing_pair[1] * audio.sample_rate)  # end time
                sample_timings.append((start, end))
            self.sample_timings = np.zeros(shape=self.wav.shape)
            for start, end in tqdm(sample_timings):
                for i in range(start, end):
                    self.sample_timings[i] = True

            self.current_start_sample = 0
        except OSError:
            self._logger.error(f"File not found: {self.wav_file}")
            raise
        except Exception:
            self._logger.error("An unknown error occurred")
            raise

    def _stream_callback(self, in_data, frame_count, time_info, status):
        self._logger.debug(
            f"Audio data with {frame_count} samples for {time_info} and status {status} received")
        start_sample = self.current_start_sample
        end_sample = start_sample + self.config.audio_config.frame_size
        try:
            played_samples = self.wav[start_sample:end_sample]
            if self.silent:
                received_samples = played_samples
            else:
                received_samples = np.fromstring(in_data, dtype=np.float32)[0::self.config.channels]
            labels = self.sample_timings[start_sample:end_sample]
            self._logger.debug(
                f"received Samples Shape {received_samples.shape} {labels.shape}")
            element = EvaluationAudioElement(
                tf.constant(received_samples),
                tf.constant(played_samples),
                tf.constant(labels)
            )
            self.buffer.add_element(element)
            self.current_start_sample += self.config.audio_config.frame_size
            return played_samples, pyaudio.paContinue
        except IndexError:
            self._logger.debug("End of evaluation wave file reached")
            return [], pyaudio.paComplete
        except Exception:
            self._logger.error("An unknown error occurred")
            raise
