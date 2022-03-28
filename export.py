import fire
import numpy as np
import soundfile
import utils

def export(pathRb: str, pathWav: str, sr: int = 22050, subtype: str = 'FLOAT'):
    """ export receiver buffer elements of a saved object file to an concatenated wave file"""
    maxBitsWav = 3.436e+10 - (12+24)*8 # headers of wav
    maxFloat32SamplesWav = maxBitsWav // 32
    receiverBuffer = utils.restoreReceiverBuffer(pathRb)
    samplesOfElement = len(receiverBuffer._buffer[0].receivedSamples)
    samplesInBuffer = len(receiverBuffer._buffer)*samplesOfElement
    maxElementsInWav = maxFloat32SamplesWav // samplesOfElement
    wavFiles = utils.roundUpDiv(samplesInBuffer, maxElementsInWav)
    receivedSampleLists = [receiverElement.receivedSamples for receiverElement in receiverBuffer._buffer]
    if wavFiles > 1:
        wavFilesSubs = [receivedSampleLists[i:i + maxElementsInWav] for i in range(0, len(receivedSampleLists), maxElementsInWav)]
        for i, e in enumerate(wavFilesSubs):
            npConcatSamples = np.concatenate(e)
            soundfile.write(pathWav.split('.wav')[0]+f'_{i}.wav', npConcatSamples, sr, subtype=subtype, format='WAV')
    else:
        npConcatSamples = np.concatenate(receivedSampleLists)
        soundfile.write(pathWav, npConcatSamples, sr, subtype=subtype, format='WAV')

def main():
    fire.Fire(export)

if __name__ == '__main__':
    main()