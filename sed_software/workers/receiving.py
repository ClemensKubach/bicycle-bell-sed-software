"""Module for receiving audio data."""

import csv
import threading
from abc import ABC, abstractmethod
from threading import Thread
import logging
import re
from typing import Optional

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_io as tfio
import pyaudio

from sed_software.data.audio.chunks import AudioChunk, ProductionAudioChunk, \
    EvaluationAudioChunk
from sed_software.data.audio.elements import AudioElement, EvaluationAudioElement
from sed_software.data.configs.configs import ReceiverConfig
from sed_software.storage.audio.persistent import AudioStorage
from sed_software.storage.audio.temporary import AudioBuffer


class AudioReceiver(Thread, ABC):
    """AudioReceiver"""

    def __init__(self, config: ReceiverConfig) -> None:
        super().__init__(daemon=True)
        self.config = config

        audio = self.config.audio_config

        self.logger = logging.getLogger(__name__)
        self.storage = AudioStorage(self.config.storage_size)
        self.buffer = self._init_buffer()

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
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
    def _init_buffer(self) -> AudioBuffer:
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
        audio_as_np_float32 = np.fromstring(in_data, np.float32)[0::self.config.channels]
        # self._stream_callback_creator(audio_as_np_float32)
        element = AudioElement(tf.constant(audio_as_np_float32))
        self.buffer.add_element(element)
        return in_data, pyaudio.paContinue

    def receive_latest_chunk(self) -> Optional[AudioChunk]:
        """return latest chunk from buffer"""
        return self.buffer.get_latest_chunk()

    @property
    def delay(self) -> float:
        """delay in seconds"""
        return self.config.audio_config.frame_size / self.config.audio_config.sample_rate

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

    def _init_buffer(self) -> AudioBuffer:
        audio = self.config.audio_config
        return AudioBuffer(ProductionAudioChunk, audio.chunk_size)


class EvaluationAudioReceiver(AudioReceiver):
    """Evaluation audio receiver"""

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
            self.logger.error(f"File not found: {self.wav_file}")
            raise
        except Exception:
            self.logger.error("An unknown error occurred")
            raise

    def _stream_callback(self, in_data, frame_count, time_info, status):
        self.logger.debug(
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
            self.logger.debug(
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
            self.logger.debug("End of evaluation wave file reached")
            return [], pyaudio.paComplete
        except Exception:
            self.logger.error("An unknown error occurred")
            raise
