"""Module containing supported types of inference models with its definitions and wrapping them
for unified usage."""

import logging
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from seds_lib.models.saved_models import BaseSavedModel


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
        return './models/converted-model.tflite'

    def _convert_model(self):
        converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model.saved_model_path)
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(self._converted_model_path, 'wb') as file:
            file.write(tflite_model)

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


class TFTensorRTModel(BaseInferenceModel):
    """TF-TensorRT Model

    Not yet supported for Windows!

    References:
        https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter
    """

    @property
    def _converted_model_path(self) -> str:
        return './models/converted-model.tftrt'

    def _convert_model(self):
        params = tf.experimental.tensorrt.ConversionParams(
            precision_mode='FP16',
            # Currently, only one engine is supported in mode INT8.
            maximum_cached_engines=1,
            use_calibration=True,
        )
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=self._converted_model_path,
            conversion_params=params,
            use_dynamic_shape=True,
            dynamic_shape_profile_strategy='Optimal',
            allow_build_at_runtime=False,
        )

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

        converter.convert(calibration_input_fn=shape_calibration_input_fn)

        # only needed, if multiple shapes should be supported
        # (Optional) Generate more TRT engines offline (same as the previous
        # option), to avoid the cost of generating them during inference.
        # def my_input_fn():
        #     for _ in range(num_runs):
        #         inp1, inp2 = ...
        #         yield inp1, inp2
        # converter.build(input_fn=my_input_fn)

        # not needed because convert() already generated one engine for our single shape
        # converter.build(input_fn=my_input_fn)

        # Save the TRT engine and the engines.
        converter.save(self._converted_model_path)

    def _prepare_interpreter(self):
        loaded_converted_model = tf.saved_model.load(self._converted_model_path,
                                                     tags=[tag_constants.SERVING])
        self.interpreter = loaded_converted_model.signatures['serving_default']

    def _predict(self, preprocessed_sample: tf.Tensor) -> float:
        batched_preprocessed_sample = tf.expand_dims(preprocessed_sample, axis=0)
        batched_result_tensor = self.interpreter(batched_preprocessed_sample)['predictions']
        result_value = batched_result_tensor[0]
        return result_value
