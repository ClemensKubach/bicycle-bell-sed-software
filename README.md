# Bicycle Bell Sound Event Detection System
**Author: Clemens Kubach**

This repository is one of three for my bachelor thesis on "Development of an Embedded System 
for Detecting Acoustic Alert Signals of Cyclists Using Neural Networks".

It contains the software as an easy-to-use and customizable CLI for the project. 
Only this part has to be installed on the target device and can be used and 
developed independently of the other components.

A trained saved model can be selected, which is then converted to an inference format 
(TFLite or TF-TRT), allowing real-time predictions to be made to a single sound event for 
live-streamed audio via connected sound devices. 

The other related repositories are:
- [bicycle-bell-sed-models](https://github.com/ClemensKubach/bicycle-bell-sed-models)
- [bicycle-bell-sed-pipeline](https://github.com/ClemensKubach/bicycle-bell-sed-pipeline)


## Getting Started
The software is based on the [PortAudio](http://www.portaudio.com/) library for audio I/O. 
Therefore, this must be installed on the system.
For more detailed installation instructions on an embedded device like the 
[Jetson Nano](#Jetson-Nano-Setup), see the corresponding chapter.

```shell
apt-get update
apt-get install portaudio19-dev

pip install --upgrade pip
pip install bicycle-bell-seds-cli

seds-cli --help
seds-cli run --tfmodel=\!crnn production
```

There are generally 4 main functionalities that are displayed with `seds-cli --help`.
- `conversion` can convert recordings of a pre-executed run command with appropriate 
parameterization for sound recording to a wave file.
- `devices` can be used for testing the available devices by doing a sound check.
- `resources` can be used for find the location of resource files like log-files or recordings.
- `run` is the main functionality of the software. 
This command is used to start a sound event detection.

## General Information
Generally, two versions of the CLI are installed: `jn-seds-cli` and `seds-cli`. 
The first one is based on the second one and only contains simplifications and specifications for 
the execution of the bicycle bell detection on the Jetson Nano. 
With the right choice of parameters, however, 
both CLIs can be used on all devices without any problems. 
Details about the differences can be found via `jn-seds-cli run --help`. In the following, the 
`jn-seds-cli` version will be used for an easier copy-and-paste usage on the Jetson Nano as 
target device.

Please use `--help` for detailed explanations of the individual software 
functionalities and parameters. 
With this you can get help for each level, i.e.: 
`jn-seds-cli --help`, `jn-seds-cli run --help`, `jn-seds-cli run evaluation --help`.

## Usage Examples
Show the location of the resources' folder:
```shell
jn-seds-cli resources where
```
Make a sound check of the audio devices:
```shell
jn-seds-cli devices soundcheck
```
Start a sound event detection with saving the logs to a file in the resources' folder and 
record the first minute of the received audio input stream:
```shell
jn-seds-cli run --tfmodel=\!crnn production --save_log=True --save_records=True --storage_length=60
```
Convert the recorded file of the previous run into a wave file:
```shell
jn-seds-cli conversion record_to_wav --path_storage_pickle="/abs/path/to/seds_cli/res/records/record-xx.pickle" --target_wav_path="./target_filepath/filename.wav"
```

### Run Command
There are two different modes with the run command: production and evaluation. 
The production mode is the main mode and receives live the current sound of the environment 
through the selected microphone device. 
The evaluation mode can play a recorded wave file with a corresponding annotation csv file and 
displays the ground-truth value as well as the prediction for the live microphone recordings.

Most parameters for the run command are available for both modes. 
Mode specific parameters can be found via `--help` for the selected mode. 
The following flags are used for the production mode, but are available for the evaluation mode too.

Select a predefined model via `--tfmodel=!model-name`, here CRNN via `!crnn`, and run the production 
mode without displaying the probability value of the predictions. The logs will be saved to a file:
```shell
jn-seds-cli run --tfmodel=\!crnn production --save_log=True --prob_logging=False
```
Use the extended YAMNet model in a production run and define that the selected (default) 
input device only has 1 input channel. Via specifying an integer for `input_device`, not `None`, 
a specific (not default) sound device can be selected:
```shell
jn-seds-cli run --tfmodel=\!yamnet_extened production --channels=1
```
Use the base YAMNet model in a production run with a lower threshold as default and activate the
logging of the probabilities to see that every prediction from a value of 0.3 will True:
```shell
jn-seds-cli run --tfmodel=\!yamnet_base production --threshold=0.3 --prob_logging=True
```

Mode specific flag-usage examples:
```shell
jn-seds-cli run --tfmodel=\!crnn evaluation --save_log=True --wav_file="/path/to/wave.wav" --annotation_file="/path/to/annotations.csv" --silent=False
```

```shell
jn-seds-cli run --tfmodel=\!crnn production --save_log=True --use_output=True
```

## Advanced Usage
Please note that in order to use the gpu, an appropriately compatible TensorFlow build must be 
installed and used with `--gpu=True`. 
In addition, the inference model type must be set to use 
TensorFlow-TenorRT via `--infer_model=tftrt`, depending on the specific machine. 
In some cases, also TFLite can be used with gpu support.
Unfortunately, TF-TRT could not yet be tested thoroughly because compatible devices or 
software dependencies were not available.
For further information, read `run --help` under the related parameters.

New trained models can be used via `--tfmodel="/path/to/tf-savedmodel-dir savedmodel=crnn`. 
The best way, is to modify the source code in the file `saved_models.py` and create a new child class of BaseSavedModel or Mono16kWaveInputSavedModel.
Then create an entry for this class in selectors.py, thus custom preprocessing and postprocessing for the model can be defined.
An easier way without modifying code is to use a saved model with the currently support interface of mono 16 kHz waveform input. 
If so, it can easily be used via `--tfmodel="/path/to/tf-savedmodel-dir --saved_model=MONO_16K_IN`.
Please note that this feature has not yet been thoroughly tested.

## Jetson Nano Setup
The following explanation based on the latest stable version of the JetPack OS (4.6.1) for the 
Jetson Nano at the time of writing. 
The use of the future version JetPack 5.0 is expected to resolve the installation issues with 
Tensorflow_io and thus possibly allow support of the GPU and Tensorflow-TRT on the Jetson Nano.
However, without development intentions, the here documented version 4.6.1 of JetPack 
should be used for now.

Make sure that JetPack 4.6.1 has been installed!
```shell
apt-get update
apt-get install portaudio19-dev

sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip

mkdir ./venv
python3 -m venv ./venv
source venv/bin/activate

pip install -U pip testresources setuptools==49.6.0 
pip install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
env H5PY_SETUP_REQUIRES=0 pip install -U --no-build-isolation h5py==3.1.0
pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 'tensorflow>=2'

python -c "from tensorflow.python.client import device_lib; device_lib.list_local_devices()"
PIP_EXTRA_INDEX_URL=https://snapshots.linaro.org/ldcg/python-cache pip install tensorflow_io
python -c "from tensorflow.python.client import device_lib; device_lib.list_local_devices()"

pip install bicycle-bell-seds-cli

jn-seds-cli --help
jn-seds-cli run --tfmodel=\!crnn production
```
Most of the installation steps for TensorFlow on the Jetson Nano are from 
"Prerequisites and Dependencies" [in corresponding the Nvidia Docs](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html).

Using Docker can save some on setup steps, but can also add some others. 
If Docker should be used on the Jetson Nano:
```shell
sudo docker run -it --rm --runtime nvidia --network host -v /home/jetson:/home/jetson --device /dev/snd nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3
```

## Development
Feel free to report bugs as issues and also contribute to the project. 
Please contact me for this. 
Especially the integration of new models and the full and tested integration of TF-TRT are still 
outstanding points of improvement. 
In addition, the SEDS-CLI will be offered completely separate from the bicycle bell sound event 
in a further step and repository.

Use the following steps to install directly from the GitHub repository and 
do not forget to call `git lfs pull` before running. 
This will download the model data.
```shell
apt-get update
apt-get install portaudio19-dev git git-lfs

git clone https://github.com/ClemensKubach/bicycle-bell-sed-software.git
cd bicycle-bell-sed-software
git lfs pull
pip install -e .
```

![system-overview](visualizations/overview.drawio.png "System Overview")
