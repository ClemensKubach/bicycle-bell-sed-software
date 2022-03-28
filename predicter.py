import time
import numpy as np
import librosa
import tensorflow as tf
import logging
import receiver



class PredicterResult(object):

    def __init__(self, probability: float, label: bool, time: float) -> None:
        self.probability = probability
        self.label = label
        self.predictionDelay = round(time*1000)/1000

class ProductionPredicterResult(PredicterResult):

    def __init__(self, predicterResult: PredicterResult) -> None:
        super().__init__(predicterResult.probability, predicterResult.label, predicterResult.predictionDelay)

class EvaluationPredicterResult(PredicterResult):

    def __init__(self, forReceived: PredicterResult, forPlayed: PredicterResult, groundTruth: PredicterResult) -> None:
        self.forReceived = forReceived
        self.forPlayed = forPlayed
        self.groundTruth = groundTruth
        self.predictionDelay = forReceived.predictionDelay





class Predicter(object):
    def __init__(self, mode, tfmodel_path, sample_rate, frame_length, hop_length, chunk_size, n_mels, window_type, top_db, threshold) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        assert mode in ('production', 'evaluation')
        self.logger.debug("Predicter initializing")
        self.mode = mode
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.n_fft = self.frame_length
        self.hop_length = hop_length
        self.chunk_size = chunk_size
        self.n_mels = n_mels

        self.window_type = window_type
        self.top_db = top_db

        self.threshold = threshold

        # Convert the model using TFLiteConverter
        tflite_model_path = './model/bicyclebell_model.tflite'
        #tf_model = tf.keras.models.load_model(tfmodel_path)
        #converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)#.from_saved_model(tfmodel_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(tfmodel_path)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter=True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.tensor_input_details = self.interpreter.get_input_details()
        self.logger.debug(f"Input Details: {str(self.tensor_input_details)}")
        self.tensor_output_details = self.interpreter.get_output_details()
        input_shape = self.tensor_input_details[0]['shape']
        self.logger.debug(f"Input Shape: {str(input_shape)}")

        self.logger.debug("Predicter initialized")

    def __samplesTo_logMelSpectrogram(self, samples):
        melspectrogram = librosa.feature.melspectrogram(y=samples, sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, window=self.window_type, hop_length=self.hop_length, dtype=np.float32)
        log_melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max, top_db=self.top_db)
        return log_melspectrogram

    def predict(self, samples) -> PredicterResult:
        time_start = time.perf_counter()
        log_melspectrogram = self.__samplesTo_logMelSpectrogram(samples)
        #self.logger.debug(f"Shape of log-melspectrogram {log_melspectrogram.shape}") # needed (batch_size=1, chunk_size, 128, 1)
        input_data = np.swapaxes(log_melspectrogram, 0, 1)
        input_data = input_data[np.newaxis, :, :, np.newaxis]
        #self.logger.debug(f"Shape of input data {input_data.shape}")
        self.interpreter.set_tensor(self.tensor_input_details[0]['index'], input_data)
        self.logger.debug("invoke prediction")
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        y_pred_prob = self.interpreter.get_tensor(self.tensor_output_details[0]['index'])[0][0] # = [[y_pred_prob]] shape=(batch_size, pred)
        y_pred_label = True if y_pred_prob > self.threshold else False
        time_end = time.perf_counter()
        return PredicterResult(y_pred_prob, y_pred_label, time_end-time_start)
        
class ProductionPredicter(Predicter):
    def __init__(self, tfmodel_path, sample_rate, frame_length, hop_length, chunk_size, n_mels, window_type='hamming', top_db=80.0, threshold=0.5) -> None:
        super().__init__('production', tfmodel_path, sample_rate, frame_length, hop_length, chunk_size, n_mels, window_type, top_db, threshold)

    def predict(self, receiverChunk: receiver.ProductionAudioReceiverChunk) -> ProductionPredicterResult:
        samples = receiverChunk.receivedSamplesChunk
        predictionOf_received = super().predict(samples)
        return ProductionPredicterResult(predictionOf_received)

class EvaluationPredicter(Predicter):
    def __init__(self, tfmodel_path, sample_rate, frame_length, hop_length, chunk_size, n_mels, window_type='hamming', top_db=80.0, threshold=0.5) -> None:
        super().__init__('evaluation', tfmodel_path, sample_rate, frame_length, hop_length, chunk_size, n_mels, window_type, top_db, threshold)

    def predict(self, receiverChunk: receiver.EvaluationAudioReceiverChunk, groundTruthLabelFunc='avg') -> EvaluationPredicterResult:
        possibleFuncs = ['avg', 'max', 'median']
        try:
            assert groundTruthLabelFunc in possibleFuncs
        except AssertionError:
            self.logger.warning(f"Selected groundTruthLabelFunc {groundTruthLabelFunc} is not accepted. Possible values are {possibleFuncs}. avg is automatically selected.")
            groundTruthLabelFunc='avg'
        
        samples = receiverChunk.receivedSamplesChunk
        predictionOf_received = super().predict(samples)

        cleanSamples = receiverChunk.playedSamplesChunk
        predictionOf_played = super().predict(cleanSamples)

        gt_prob = np.average(receiverChunk.labelsChunk)
        if groundTruthLabelFunc == 'avg':
            gt_label = True if gt_prob >= 0.5 else False
        elif groundTruthLabelFunc == 'max':
            gt_label = True if np.max(receiverChunk.labelsChunk) == 1 else False
        elif groundTruthLabelFunc == 'median':
            gt_label = True if np.median(receiverChunk.labelsChunk) == 1 else False
        ground_truth = PredicterResult(gt_prob, gt_label, 0.0)
        return EvaluationPredicterResult(predictionOf_received, predictionOf_played, ground_truth)