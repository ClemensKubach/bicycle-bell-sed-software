"""Inference Models"""
import logging
from abc import ABC, abstractmethod

import tensorflow as tf

from saved_models import BaseSavedModel


class BaseInferenceModel(ABC):
    """Base Inference Model"""

    def __init__(self, saved_model: BaseSavedModel) -> None:
        self.logger = logging.getLogger(__name__)
        self.saved_model = saved_model
        self._convert_model()
        self._prepare_interpreter()

    @property
    @abstractmethod
    def _converted_model_path(self) -> str:
        """path to converted models"""

    @abstractmethod
    def _convert_model(self):
        """converts Model into inference models"""

    @abstractmethod
    def _prepare_interpreter(self):
        """prepare inference models interpreter"""

    @abstractmethod
    def _predict(self, preprocessed_sample) -> tf.Tensor:
        """predict with models interpreter. returns tensor with prediction result for one sample,
        not batch. """

    def inference(self, sample: tf.Tensor, sample_rate: int):
        """inference"""
        preprocessed_sample = self.saved_model.preprocess(sample, sample_rate)
        inference_result = self._predict(preprocessed_sample)
        prob, label = self.saved_model.extract_prediction(inference_result)
        return prob, label


class TFLiteInferenceModel(BaseInferenceModel):
    """TFLite Inference Model"""

    @property
    def _converted_model_path(self) -> str:
        return './models/bicycle_bell_model.tflite'

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
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.tensor_input_details = self.interpreter.get_input_details()
        print('CLEMENS', str(self.tensor_input_details))
        self.logger.debug(f"Input Details: {str(self.tensor_input_details)}")
        self.tensor_output_details = self.interpreter.get_output_details()
        input_shape = self.tensor_input_details[0]['shape']
        self.logger.debug(f"Input Shape: {str(input_shape)}")

    def _predict(self, preprocessed_sample: tf.Tensor) -> tf.Tensor:
        batched_preprocessed_sample = tf.expand_dims(preprocessed_sample, axis=0)
        print(batched_preprocessed_sample)
        self.interpreter.set_tensor(self.tensor_input_details[0]['index'],
                                    batched_preprocessed_sample)
        self.logger.debug("invoke prediction")
        self.interpreter.invoke()
        # = [[y_pred_prob]] shape=(batch_size, pred)
        result_tensor = self.interpreter.tensor(self.tensor_output_details[0]['index'])[0]
        return result_tensor
