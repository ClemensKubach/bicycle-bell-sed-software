# thesis-embedded-sed-deployment

**Author: Clemens Kubach**

This repository includes the software for embedding the neural network model in an executable system and deploy it on a device.
It is a part of the results of my bachelor thesis [], as well as the git repository [https://github.com/ClemensKubach/thesis-embedded-sed.git] for the development of the neural network model.



## Getting Started
- clone git repo `https://github.com/ClemensKubach/thesis-embedded-sed-deployment.git`
- Change current directory `cd thesis-embedded-sed-deployment`
- install python3 and pip (dev in Python 3.9.7)
  - for jetson nano with JetPack 4.6:
    - `sudo apt update`
    - `sudo apt install python3-pip`
    - `sudo apt-get install portaudio19-dev python-pyaudio` for pyaudio
- install dependencies with: `pip3 install -r requirements.txt`
  - if `echo $JP_VERSION` is unset, specify it with your JetPack version: here `export JP_VERSION=461`
  - `sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow` for [latest TF on Jetson Nano](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
- run software: `python3 embedded-sed-system.py "./model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05/"`


## Configurations

### Download Tensorflow Model from GOogle CLoud Storage
In the following example, the model `model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05` will be downloaded from Google Cloud Storage.
1. `gcloud auth login`
2. `gsutil -m cp "gs://clemens-thesis-dataset-bucket/FSD50K/model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05.json" ./model/`
3. `gsutil -m cp "gs://clemens-thesis-dataset-bucket/FSD50K/model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05.json" ./model/`

### Download System Test Files from Google Cloud Storage
1. `gcloud auth login`
2. `gsutil -m cp -r "gs://clemens-thesis-dataset-bucket/FSD50K/finalSystemTest" ./systemtest/`

### Configurate embedded-sed-system Script Parameter
Generic example with basic settings: 

```python embedded-sed-system.py "./model/modelName" --mode="production" --silent=False --bufferMaxSize=-1 --inputDevice_index=None --outputDevice_index=None --wavFilepath=None --annotationFilepath=None --input=True --output=False --channels=1 --sample_rate=22050 --frame_size=0.046 --hop_size=0.023 --chunk_size=42 --n_mels=128 --gpu=False --loglevel="info" --logProb=False```

#### Production Mode
Minimal example with using the basic config: 

```python embedded-sed-system.py "./model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05/"```

Example for using GPU and restricting the buffer size to holding samples of over an hour:

```python embedded-sed-system.py "./model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05/" --gpu=True --bufferMaxSize=100000```

Example for debugging and advanced logging without saving records:

```python embedded-sed-system.py "./model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05/" --loglevel="debug" --logProb=True --saveRecords=False```

#### Evaluation Mode
Example for evaluating the evaluation wave file with the live sound environment noises and microphone influences.

```python embedded-sed-system.py "./model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05/" --mode='evaluation' --output=True --silent=False --wavFilepath='systemtest\finalSystemTest\systemEval_concat_selected.wav' --annotationFilepath='systemtest\finalSystemTest\timings_annotation_systemEval.csv'```

Example for evaluating the given systemtest wave file without receiving it with a microphone. Instead the wave file data is used directly as model input. The evaluation wave file will be played just for the user.

```python embedded-sed-system.py "./model/model_dev2_filewise_withoutSeperateConvNets_withoutSeperateFcNets_withoutLrSchedule_1e-05/" --mode='evaluation' --output=True --silent=True --wavFilepath='systemtest\finalSystemTest\systemEval_concat_selected.wav' --annotationFilepath='systemtest\finalSystemTest\timings_annotation_systemEval.csv'```

#### All Parameters
The following paramters are defining the selected mode or are directly dependant of it.
- `mode` - Their are two possible modes: `'production'` and `'evaluation'`. The evaluation mode offers a comparision of prepared wave files with corresponding ground truth annotations.
- `silent` - Only relevant for the evaluation mode. An evaluation can be invoked by playing an example wave file through the sound output device `output=True` and receives it again if `silent=False`. This option has the advantage of evaluating the whole hardware system including the influence of the given microphone. With `silent=False`, no input device is needed. The evaluation is only determinded for the model.
- `input` - True/False specifying the usage of an input sound device. Only *not necessary* for the `silent=True` `evaluation` mode.
- `output` - True/False specifying the usage of an output sound device. *Necessary* just for the `silent=False` `evaluation` mode. Interesting for debugging purpose.
- `bufferMaxSize` - Describes the maximum number of AudioReceiverElement objects in the buffer at the same time. The default value is -1 for keeping all received examples in the memory. For devices with low memory capacities, a high number should be selected instead.
- `wavFilepath` - Only used for the evaluation mode. The filepath to the wave systemtest file.
- `annotationFilepath` - Only used for the evaluation mode. The filepath to the annotations csv file for the selected corresponding wave systemtest file.

The following paramters should be selected for the given devices. In general, a sample rate of 22050 should be enough.
- `inputDevice_index` - Index number of a input sound device. Default is None, for using the selected default device.
- `outputDevice_index` - Index number of a output sound device. Default is None, for using the selected default device.
- `channels` - number channels of the sound input. Currently, only the first channel will be supported for the model prediction.
- `gpu` - should be true, if an gpu is available in the given system.
- `saveRecords` - For saving the recorded or received sound input. Default is true, if less storage is available false should be used.
- `sample_rate` - number of samples per second

Because the following paramters are dependant from the architecture of the trained model, they should stay unmodified in most cases.
- `frame_size` - seconds of one frame
- `hop_size` - seconds of the skip size for the continous sliding window over the samples
- `chunk_size` - number of frames in a chunk of samples
- `n_mels` - number of mels

Parameters for the logging configuration.
- `loglevel` in most cases `"info"` or `"debug"`.
- `logProb` Option to log the exact probability values of every prediction, instead of just output the hard thresholded label True/False.


## Architecture
- `embedded-sed-system.py` - Script for the initialization of the predicter and receiver objects. Run this from command line with individual config for the defined parameters. 
- `predicter.py` - Includes classes and implementations for the prediction process. The prediction function of a Predicter objects expects an AudioReceiverChunk object and returns an PredicterResult object.
  - Predicter <- ProductionPredicter, EvaluatationPredicter
  - PredicterResult <- ProductionPredicterResult, EvaluationPredicterResult
- `receiver.py` - Includes classes and implementations for the receiving process of the audio samples. Their data will be later used for the prediction. The received buffer can be saved in storage for later reusablity.
  - AudioReceiver(AudioReceiverBuffer)                                  <- ProductionAudioReceiver, EvaluationAudioReceiver
  - AudioReceiverBuffer(AudioReceiverElement) --> AudioReceiverChunk    <- ProductionAudioReceiverBuffer, EvaluationAudioReceiverBuffer
  - AudioReceiverChunk(AudioReceiverElement)                            <- ProductionAudioReceiverChunk, EvaluationAudioReceiverChunk
  - AudioReceiverElement                                                <- ProductionAudioReceiverChunk, EvaluationAudioReceiverChunk
- `utils.py` - Helper functions.
- `export.py` - Extract the received audio samples from a backup of an previous AudioReceiverBuffer object to a wave file.
- logs - Target directory of all logging outputs as file in parallel to the console prints.
- model - Directory for the downloaded and trained tensorflow models. Contains the dynamically created tflite model compilation.
- pyaudio - Contains some example scripts for an easy testing of sound devices and depended settings. From [https://people.csail.mit.edu/hubert/pyaudio/].
- records - Target location for the automatically saved AudioReceiverBuffer objects.
- systemtest - Contains system test wave files and annotation csv files for the EvaluationPredicter.


