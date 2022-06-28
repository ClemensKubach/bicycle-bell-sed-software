"""Module for receivers."""

import csv
import threading
import time
from abc import ABC, abstractmethod
from threading import Thread
import logging
from typing import Optional, Union, Any, Tuple
from random import randint

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import pyaudio

from seds_cli.seds_lib.data.audio.chunks import ProductionAudioChunk
from seds_cli.seds_lib.data.audio.chunks import EvaluationAudioChunk
from seds_cli.seds_lib.data.audio.elements import AudioElement, ProductionAudioElement
from seds_cli.seds_lib.data.audio.elements import EvaluationAudioElement
from seds_cli.seds_lib.data.configs.configs import ReceiverConfig
from seds_cli.seds_lib.data.time.delay import ReceiverDelay
from seds_cli.seds_lib.storage.audio.persistent import AudioStorage
from seds_cli.seds_lib.storage.audio.temporary import AudioBuffer

# pylint: disable=too-many-instance-attributes
from seds_cli.seds_lib.utils import audio_maths


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

    # pylint: disable=unused-argument
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

        element, output_data, next_status = self._create_audio_element(in_data)

        self.buffer.add_element(element)
        self.storage.add_element(element)
        with self._lock:
            if self._measure_time and start_time is not None:
                end_time = time.perf_counter()
                self._stream_callback_time = end_time - start_time
                self._measure_time = False
        self._logger.debug(f'Evaluation output_data size: {output_data.shape[0]}')
        return output_data, next_status

    @abstractmethod
    def _create_audio_element(self, in_data: Any) -> Tuple[Optional[AudioElement], Any, int]:
        """de-serialize input data and create AudioElement."""
        audio_as_np_float32 = np.fromstring(in_data, np.float32)[0::self.config.channels]
        element = AudioElement(tf.constant(audio_as_np_float32))
        next_status = pyaudio.paContinue
        return element, in_data, next_status

    def receive_latest_chunk(self) -> Optional[Union[ProductionAudioChunk,
                                                     EvaluationAudioChunk]]:
        """Returns the latest chunk from the buffer. Should be called from another thread."""
        chunk = self.buffer.extract_latest_chunk()
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
        return AudioBuffer(ProductionAudioChunk, self.config.audio_config)

    def _create_audio_element(self, in_data: Any) -> Tuple[Optional[AudioElement], Any, int]:
        """de-serialize input data and create AudioElement."""
        audio_as_np_float32 = np.fromstring(in_data, np.float32)[0::self.config.channels]
        element = ProductionAudioElement(tf.constant(audio_as_np_float32))
        next_status = pyaudio.paContinue
        return element, in_data, next_status


class EvaluationAudioReceiver(AudioReceiver):
    """Evaluation mode AudioReceiver as Thread.
    Expects a ReceiverConfig instance.

    Loads the wav_file and the corresponding annotation_file.
    If `silent=True`, the value for received is equal to played. To still have the opportunity
    to use an output device for playing the audio,
    the evaluation process will continue to play and compute in real time.

    It initializes an AudioStorage and AudioBuffer instance. Then PortAudio via PyAudio is used
    for opening a stream such that callbacks for each frame can be called asynchronous.

    References:
        https://people.csail.mit.edu/hubert/pyaudio/
    """

    def _init_buffer(self) -> AudioBuffer:
        return AudioBuffer(EvaluationAudioChunk, self.config.audio_config)

    # pylint: disable=too-many-locals
    def __init__(self, config: ReceiverConfig,
                 wav_file: str, annotation_file: str, silent: bool) -> None:
        super().__init__(config)
        audio = self.config.audio_config
        try:
            self.wav_file = wav_file
            self.annotation_file = annotation_file
            self.silent = silent

            self._logger.info('Reading wave and csv file... (may take a few seconds)')
            wav_audio, in_sample_rate = tf.audio.decode_wav(
                tf.io.read_file(self.wav_file),
                desired_channels=1,
                desired_samples=-1,
            )
            self.wav = tf.squeeze(wav_audio)
            if int(in_sample_rate) != int(audio.sample_rate):
                self.wav = tfio.audio.resample(
                    self.wav,
                    in_sample_rate,
                    audio.sample_rate
                )
                # if tf bug about necessary (unused) tensorflow_io import is resolved, use resampy:
                # self.wav = resampy.resample(tf.squeeze(wav_audio), sr, audio.sample_rate)
            self.annotations = []
            with open(annotation_file, 'r', newline='', encoding='utf-8') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    # row = (start_time, end_time, file_name, annotation_bool)
                    if bool(row[3] == 'True'):
                        self.annotations.append((float(row[0]), float(row[1])))
            sample_timings = []
            for start_time_sec, end_time_sec in self.annotations:
                start_sample = audio_maths.seconds_to_samples(start_time_sec, audio.sample_rate)
                end_sample = audio_maths.seconds_to_samples(end_time_sec, audio.sample_rate)
                sample_timings.append((start_sample, end_sample))
            self.sample_timings = np.zeros(shape=self.wav.shape)
            for start, end in sample_timings:
                for i in range(start, end):
                    # set the truth value for the sample timing to True
                    # because of frame_size, some (<frame_size) last samples can get lost
                    if i < len(self.sample_timings):
                        self.sample_timings[i] = True
            self.current_start_sample = 0
        except OSError:
            self._logger.error(f"File not found: {self.wav_file}")
            raise
        except Exception:
            self._logger.error("An unknown error occurred")
            raise

    def _create_audio_element(self, in_data: Any) -> Tuple[Optional[AudioElement], Any, int]:
        """de-serialize input data and create AudioElement."""
        self._logger.debug(f'Received data string size {len(in_data)}')
        try:
            start_sample = self.current_start_sample
            end_sample = start_sample + self.config.audio_config.frame_size
            played_samples = self.wav[start_sample:end_sample]
            if self.silent:
                received_samples = played_samples
            else:
                audio_as_np_float32 = np.fromstring(in_data, np.float32)[0::self.config.channels]
                received_samples = audio_as_np_float32
            self._logger.debug(f'Received number of samples: {received_samples.shape[0]}')
            labels = self.sample_timings[start_sample:end_sample]
            if end_sample >= self.wav.shape[0]:
                self._logger.error('Evaluation callback index error!')
                raise IndexError
            element = EvaluationAudioElement(
                tf.constant(received_samples),
                tf.constant(played_samples),
                tf.constant(labels)
            )
            self.current_start_sample += self.config.audio_config.frame_size
            next_status = pyaudio.paContinue
            return element, played_samples, next_status
        except IndexError:
            self._logger.info("End of evaluation wave file reached")
            next_status = pyaudio.paComplete
            self._logger.info('Press Ctrl+C to terminate!')
            return None, b'', next_status
        except Exception:
            self._logger.error("An unknown error occurred")
            raise
