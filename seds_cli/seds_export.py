"""tool for exporting content of audio storage file to wave files"""

import logging

import fire
import tensorflow as tf

from seds_cli.seds_lib.utils import audio_maths
from seds_cli.seds_lib.utils import file_utils


def main(path_storage_pickle: str, target_wav_path: str, sample_rate: int = 16000):
    """ export elements of a saved audio storage file to a concatenated wave file """
    _logger = logging.getLogger(__name__)
    max_bits_wav = 3.436e+10 - (12 + 24) * 8  # headers of wav
    max_float32_samples_wav = max_bits_wav // 32
    storage = file_utils.restore_audio_storage(path_storage_pickle)

    storage_elements = storage.get_elements()
    samples_of_element = len(storage_elements[0].received_samples)
    samples_in_buffer = len(storage_elements) * samples_of_element

    max_elements_in_wav = int(max_float32_samples_wav // samples_of_element)
    wav_files = audio_maths.round_up_div(samples_in_buffer, max_elements_in_wav)
    _logger.info(f'Creating {wav_files} wave files...')
    received_sample_lists = [element.received_samples for element in storage_elements]
    if wav_files > 1:
        wav_files_subs = [received_sample_lists[i:i + max_elements_in_wav]
                          for i in range(0, len(received_sample_lists), max_elements_in_wav)
                          ]
        for i, element in enumerate(wav_files_subs):
            concat_samples = tf.concat(element, 0)
            tensor_string = tf.audio.encode_wav(tf.expand_dims(concat_samples, -1), sample_rate)
            tf.io.write_file(target_wav_path.split('.wav')[0] + f'_{i}.wav', tensor_string)
    else:
        concat_samples = tf.concat(received_sample_lists, 0)
        tensor_string = tf.audio.encode_wav(tf.expand_dims(concat_samples, -1), sample_rate)
        tf.io.write_file(target_wav_path, tensor_string)


if __name__ == '__main__':
    fire.Fire(main)
