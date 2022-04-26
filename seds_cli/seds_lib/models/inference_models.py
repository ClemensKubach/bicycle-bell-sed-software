"""Module containing supported types of inference models with its definitions and wrapping them
for unified usage."""

import logging
import os.path
from abc import ABC, abstractmethod

import tensorflow as tf

from seds_cli.seds_lib.models.saved_models import BaseSavedModel


class BaseInferenceModel(ABC):
    """Abstract base inference model class."""

    def __init__(self, saved_model: BaseSavedModel, window_size, batch_size: int = 1) -> None:
        """Expects an instance of a BaseSavedModel implementation, the window_size as
        number of samples and the batch_size (should be kept at 1).
        """
        self._logger = logging.getLogger(__name__)
        self.saved_model = saved_model
        self.window_size = window_size
        self.batch_size = batch_size
        self._convert_model()
        self._prepare_interpreter()

    @property
    @abstractmethod
    def _converted_model_path(self) -> str:
        """path to the converted model"""

    @abstractmethod
    def _convert_model(self):
        """defines how the model will be converted into a performant inference model"""

    @abstractmethod
    def _prepare_interpreter(self):
        """defines how to prepare the inference interpreter"""

    @abstractmethod
    def _predict(self, preprocessed_sample) -> float:
        """Defines how to predict with models' interpreter on the given preprocessed tensor and
        returns a single probability."""

    def inference(self, sample: tf.Tensor, sample_rate: int):
        """Pipes the samples' tensor through preprocessing, inference and extracts the prediction
        value as unified probability-label tuple."""
        preprocessed_sample = self.saved_model.preprocess(sample, sample_rate)
        inference_result = self._predict(preprocessed_sample)
        prob, label = self.saved_model.extract_prediction(inference_result)
        return prob, label


class TFLiteInferenceModel(BaseInferenceModel):
    """
    TFLite Inference Model

    References:
        https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
    """

    @property
    def _converted_model_path(self) -> str:
        return os.path.join(self.saved_model.saved_model_path, 'converted-model.tflite')

    def _convert_model(self):
        if os.path.exists(self._converted_model_path):
            self._logger.info('The corresponding saved model was already converted into the '
                              'desired inference format in a previous execution. '
                              'If frame and window configuration is not changed, '
                              'it can be used seamlessly without re-converting.')
            keep_converted_answer = input('> Use previously converted tflite model? (Y/n)\n')
            if keep_converted_answer.lower() == 'y':
                self._logger.info(f'Existing model {self._converted_model_path} will be used.')
                return

        self._logger.info('Converting model ... (may take a few minutes)')
        converter = tf.lite.TFLiteConverter.from_saved_model(
            self.saved_model.saved_model_path,
        )
        # converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(self._converted_model_path, 'wb') as file:
            file.write(tflite_model)
        self._logger.info('Converted model of inference type Tensorflow Lite was created and '
                          f'saved in {self._converted_model_path}.')

    def _prepare_interpreter(self):
        # Load the TFLite models and allocate tensors.
        self.interpreter = tf.lite.Interpreter(self._converted_model_path)
        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        input_shape = self.tensor_input_details[0]['shape']
        self.interpreter.resize_tensor_input(self.tensor_input_details[0]['index'],
                                             (self.batch_size, self.window_size),
                                             strict=True)
        # Get input and output tensors.
        self._logger.debug(f"Input Details: {str(self.tensor_input_details)}")

        self._logger.debug(f"Input Shape: {str(input_shape)}")
        self.interpreter.allocate_tensors()

    def _predict(self, preprocessed_sample: tf.Tensor) -> float:
        batched_preprocessed_sample = tf.expand_dims(preprocessed_sample, axis=0)
        self.interpreter.set_tensor(self.tensor_input_details[0]['index'],
                                    batched_preprocessed_sample)
        self.interpreter.invoke()
        # = [[y_pred_prob]] shape=(batch_size, pred)
        result_value = self.interpreter.get_tensor(self.tensor_output_details[0]['index'])[0]
        return result_value


# pylint:disable=unexpected-keyword-arg
# pylint:disable=no-member
class TFTensorRTModel(BaseInferenceModel):
    """TF-TensorRT Model (memory sizes for Jetson Nano (4GB) selected)

    Tensorflow v2.7.0 and v2.8.0.

    Not yet supported for Windows!

    Currently, the predictor tries to access the value of class_output if it is
    defined as output layer. If not, the first model output will be used.
    Further (multiple) model outputs will be ignored.

    References:
        https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter
    """

    def __init__(self, saved_model: BaseSavedModel, window_size):
        super().__init__(saved_model, window_size)
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )

    @property
    def _converted_model_path(self) -> str:
        return os.path.join(self.saved_model.saved_model_path, 'converted-model.tftrt')

    def _convert_model(self):
        self._logger.info('You are trying to use TF-TRT as inference model. This is only '
                          'supported for UNIX machines with either tensorflow 2.7.0 or 2.8.0'
                          ' installed.')

        if os.path.exists(self._converted_model_path):
            self._logger.info('The corresponding saved model was already converted into the '
                              'desired inference format in a previous execution. '
                              'If frame and window configuration is not changed, '
                              'it can be used seamlessly without re-converting.')
            keep_converted_answer = input('> Use previously converted tftrt model? (Y/n)\n')
            if keep_converted_answer.lower() == 'y':
                self._logger.info(f'Existing model {self._converted_model_path} will be used.')
                return

        self._logger.info('Converting model ... (may take a few minutes)')
        try:
            params = tf.experimental.tensorrt.ConversionParams(
                precision_mode='FP16',
                # Currently, only one engine is supported in mode INT8.
                maximum_cached_engines=1,
                use_calibration=True,
                allow_build_at_runtime=True,
                max_workspace_size_bytes=(1 << 30),
            )
            converter = tf.experimental.tensorrt.Converter(
                input_saved_model_dir=self.saved_model.saved_model_path,
                conversion_params=params,
                use_dynamic_shape=True,
                dynamic_shape_profile_strategy='Optimal',
            )
        except ValueError:
            self._logger.error('You tried to use TF-TRT as inference model as failed. This is '
                               'probably because of a version mismatch of tensorflow (has to be '
                               'either v2.7.0 or v2.8.0) or you are trying to run this on a '
                               'Windows machine. ')
            raise

        # Define a generator function that yields input data, and run INT8
        # calibration with the data. All input data should have the same shape.
        # At the end of convert(), the calibration stats (e.g. range information)
        # will be saved and can be used to generate more TRT engines with different
        # shapes. Also, one TRT engine will be generated (with the same shape as
        # the calibration data) for save later.
        def shape_calibration_input_fn():
            for _ in range(1):
                input_shapes = [(self.batch_size, self.window_size)]
                yield [tf.zeros(shape, tf.float32) for shape in input_shapes]

        # if INT8
        # converter.convert(calibration_input_fn=shape_calibration_input_fn)
        self._logger.debug('TF-TRT - Start conversion')
        converter.convert()
        self._logger.debug('TF-TRT - Conversion finished')

        # only needed, if multiple shapes should be supported
        # (Optional) Generate more TRT engines offline (same as the previous
        # option), to avoid the cost of generating them during inference.
        # def my_input_fn():
        #     for _ in range(num_runs):
        #         inp1, inp2 = ...
        #         yield inp1, inp2
        # converter.build(input_fn=my_input_fn)

        self._logger.debug('TF-TRT - Start building')
        converter.build(input_fn=shape_calibration_input_fn)
        self._logger.debug('TF-TRT - Building finished')

        # Save the TRT engine and the engines.
        self._logger.debug('TF-TRT - Save engines')
        converter.save(self._converted_model_path)
        self._logger.debug('TF-TRT - Engines saved')
        self._logger.info('Converted model of inference type Tensorflow TRT was created and '
                          f'saved in {self._converted_model_path}.')

    def _prepare_interpreter(self):
        self._logger.debug('TF-TRT - Load TRT model')
        loaded_converted_model = tf.saved_model.load(self._converted_model_path)
        self._logger.debug('TF-TRT - TRT model loaded')
        self._logger.debug('TF-TRT - Load model graph_func')
        self.interpreter = loaded_converted_model.signatures['serving_default']
        self._logger.debug('TF-TRT - model graph_func loaded')

    def _predict(self, preprocessed_sample: tf.Tensor) -> float:
        batched_preprocessed_sample = tf.expand_dims(preprocessed_sample, axis=0)
        batched_result_dict = self.interpreter(batched_preprocessed_sample)
        # Currently, the predictor tries to access the value of class_output if it is
        # defined as output layer. If not, the first model output will be used.
        # Further (multiple) model outputs will be ignored.
        if 'class_output' in batched_result_dict:
            target_key = 'class_output'
        else:
            target_key = next(iter(batched_result_dict))
        batched_result_tensor = batched_result_dict[target_key]
        result_value = batched_result_tensor[0]
        return result_value
