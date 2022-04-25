"""Module with all data classes about delays"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceiverDelay:
    """Data about the delay concerning the receiver worker.

    Attributes

    callback_offset_estimated_time (float)
        in seconds, needed to process the callback
    """
    callback_offset_estimated_time: float

    @property
    def delay(self):
        """Time that is relevant for the delay between input and output of the overall system.

        Notes:
            callback_offset_estimated_time:
                Processed asynchronously, thus only relevant once for the latest input element in
                the chunk

        Returns:
            callback_offset_estimated_time
        """
        return self.callback_offset_estimated_time


@dataclass(frozen=True)
class ChunkDelay:
    """Data about the delay concerning the chunk.

    Attributes

    processing_time (float)
        in seconds, needed mainly for the concatenation of the elements

    max_in_buffer_waiting_time (float)
        in seconds, time the oldest element in the buffer waited until chunk processing
    """
    processing_time: float
    max_in_buffer_waiting_time: float

    @property
    def delay(self):
        """Time that is relevant for the delay between input and output of the overall system.

        Notes:
            processing_time:
                Elements are prepared by the Predictor thread to form an AudioChunk

            max_in_buffer_waiting_time:
                Should be used for more detailed interval specification of the delay.
                Including this time is only valid the worst-case.
                This margin can be lowered by parallel Predictor processes (NotYetImplemented).

        Returns:
            processing_time + max_in_buffer_waiting_time
        """
        return self.processing_time + self.max_in_buffer_waiting_time


@dataclass(frozen=True)
class PredictorDelay:
    """Data about the delay within the Predictor Thread.

    Attributes

    chunk_delay (ChunkDelay)
        delay object concerning the chunk

    inference_time (float)
        in seconds, time needed for running an inference_time step on the model,
        including preprocessing
    """
    chunk_delay: ChunkDelay
    inference_time: float

    @property
    def delay(self):
        """Time that is relevant for the delay between input and output of the overall system.

        Notes:
            chunk_delay.delay:
                Overall relevant chunk delay

            inference_time:
                Time needed of the Predictor Thread for getting a result
                for the current chunk/window.

        Returns:
            relevant_delay_of(chunk_delay) + inference_time
        """
        return self.chunk_delay.delay + self.inference_time


@dataclass(frozen=True)
class Delay:
    """Data about the delay of all parts of the system.

    Attributes

    receiving_delay (ReceiverDelay)
        delay object concerning the Receiver (Thread)

    predicting_delay (PredictorDelay)
        delay object concerning the Predictor (Thread)
    """
    receiving_delay: ReceiverDelay
    predicting_delay: PredictorDelay

    @property
    def delay(self):
        """Overall time that is relevant for the delay between input and output of the
        overall system.

        Notes:
            receiving_delay.delay:
                Overall relevant receiving_delay delay

            predicting_delay.delay:
                Overall relevant predicting_delay delay

        Returns:
            relevant_delay_of(receiving_delay) + relevant_delay_of(predicting_delay)
        """
        return self.receiving_delay.delay + self.predicting_delay.delay
