"""export tool"""

import fire
import numpy as np
import soundfile
import utils

def main(path_rb: str, path_wav: str, sr: int = 22050, subtype: str = 'FLOAT'):
    """ export receiver buffer elements of a saved object file to an concatenated wave file"""
    max_bits_wav = 3.436e+10 - (12+24)*8 # headers of wav
    max_float32_samples_wav = max_bits_wav // 32
    receiver_buffer = utils.restore_receiver_storage(path_rb)
    samples_of_element = len(receiver_buffer._buffer[0].receivedSamples)
    samples_in_buffer = len(receiver_buffer._buffer)*samples_of_element
    max_elements_in_wav = max_float32_samples_wav // samples_of_element
    wav_files = utils.round_up_div(samples_in_buffer, max_elements_in_wav)
    received_sample_lists = [receiverElement.receivedSamples for receiverElement in receiver_buffer._buffer]
    if wav_files > 1:
        wav_files_subs = [received_sample_lists[i:i + max_elements_in_wav] for i in range(0, len(received_sample_lists), max_elements_in_wav)]
        for i, e in enumerate(wav_files_subs):
            np_concat_samples = np.concatenate(e)
            soundfile.write(path_wav.split('.wav')[0]+f'_{i}.wav', np_concat_samples, sr, subtype=subtype, format='WAV')
    else:
        np_concat_samples = np.concatenate(received_sample_lists)
        soundfile.write(path_wav, np_concat_samples, sr, subtype=subtype, format='WAV')


if __name__ == '__main__':
    fire.Fire(main)