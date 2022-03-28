from threading import Thread
import threading
import fire
from predicter import EvaluationPredicter, EvaluationPredicterResult, ProductionPredicterResult, ProductionPredicter
from receiver import EvaluationAudioReceiver, EvaluationAudioReceiverBuffer, ProductionAudioReceiverBuffer, ProductionAudioReceiver
import utils
import keyboard
import logging
import logging.handlers
import logging.config
import time
import os

def execute(tfmodel_path, 
            mode='production', 
            silent=False, 
            input=True,
            output=False,
            bufferMaxSize=-1, 
            wavFilepath=None,
            annotationFilepath=None,
            
            inputDevice_index=None, 
            outputDevice_index=None, 
            channels=1, 
            gpu=False,
            saveRecords=True,
            sample_rate=22050,

            frame_size=0.046,
            hop_size=0.023,
            chunk_size=42,
            n_mels=128,
            
            loglevel="info",
            logProb=False ):

    logger = logging.getLogger()
    console = logging.StreamHandler()
    if loglevel.lower() == "info":
        logger.setLevel(logging.INFO)
        console.setLevel(logging.INFO)
        loggerFormat = '%(asctime)s %(levelname)s %(module)s: %(message)s'
        seperateDelayLog = False
    elif loglevel.lower() == "debug":
        logger.setLevel(logging.DEBUG)
        console.setLevel(logging.DEBUG)
        loggerFormat = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
        seperateDelayLog = True
    else:
        logger.setLevel(logging.DEBUG)
        console.setLevel(logging.DEBUG)
        loggerFormat = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
        seperateDelayLog = True
    
    #q = Queue()
    #lp = threading.Thread(target=logger_thread, args=(q,), daemon=True)
    #qh = logging.handlers.QueueHandler(q)
    timestamp = time.strftime(f'%Y.%m.%d-%H.%M.%S')
    logging.basicConfig(format=loggerFormat, filename=f'./logs/{timestamp}.log')
    formatter = logging.Formatter(loggerFormat)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

    envvar = "CUDA_VISIBLE_DEVICES"
    if gpu:
        os.environ[envvar] = "0"
        logger.info(f'Environment variable {envvar} is set to 0')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info(f'Environment variable {envvar} is set to -1')

    try:
        assert mode == 'production' or (mode == 'evaluation' and wavFilepath is not None and annotationFilepath is not None)
    except AssertionError:
        raise

    logger.debug("System execution is starting...")

    frame_length = utils.get_frameLength(frame_size, sample_rate)
    hop_length = utils.get_hopLength(hop_size, sample_rate)

    if mode == 'production':
        logger.info(f'Production mode selected')

        predicter = ProductionPredicter(
                        tfmodel_path=tfmodel_path,
                        sample_rate=sample_rate, 
                        frame_length=frame_length, 
                        hop_length=hop_length, 
                        chunk_size=chunk_size,
                        n_mels=n_mels, 
                        window_type='hamming', 
                        top_db=80 
                    )

        receiver = ProductionAudioReceiver(
            buffer=ProductionAudioReceiverBuffer(bufferMaxSize=bufferMaxSize, chunk_size=chunk_size, frame_length=frame_length, hop_length=hop_length),
            sample_rate=sample_rate, 
            frame_length=frame_length, 
            hop_length=hop_length,
            channels=channels
        )
    elif mode == 'evaluation':
        silentTxt = 'with' if silent else 'without'
        logger.info(f'Evaluation mode selected {silentTxt} using the direct input of the wave file')

        predicter = EvaluationPredicter(
                        tfmodel_path=tfmodel_path,
                        sample_rate=sample_rate, 
                        frame_length=frame_length, 
                        hop_length=hop_length, 
                        chunk_size=chunk_size,
                        n_mels=n_mels, 
                        window_type='hamming', 
                        top_db=80 
                    )

        receiver = EvaluationAudioReceiver(
            buffer=EvaluationAudioReceiverBuffer(bufferMaxSize=bufferMaxSize, chunk_size=chunk_size, frame_length=frame_length, hop_length=hop_length),
            sample_rate=sample_rate, 
            frame_length=frame_length, 
            hop_length=hop_length,
            channels=channels,
            wavFilepath=wavFilepath, 
            annotationFilepath=annotationFilepath, 
            silent=silent
        )
        
    receiver.initReceiver(input=input, output=output, inputDevice_index=inputDevice_index, outputDevice_index=outputDevice_index)
    
    try:
        chunkTime = utils.chunkTime(hop_size, chunk_size)
        logger.debug("Worker started")
        logger.info('Press Ctrl+C or Interrupt the Kernel')  
        while True:
            time.sleep(0.00001)
            if keyboard.is_pressed('q'):
                raise KeyboardInterrupt
            elif keyboard.is_pressed('s'):
                tSynch = Thread(target=nonBlocking_delayPrint, args=(delayForSync, logger), daemon=False)
                tSynch.start()
                logger.warning("Synchronization visualization started")
            chunk_sampleFrames = receiver.receiveNextChunkOfBuffer()
            if chunk_sampleFrames is not None:
                y = predicter.predict(chunk_sampleFrames)
                delayForSync = receiver.getDelay(seperate=False) + y.predictionDelay
                delayPrint = receiver.getDelay(seperateDelayLog)
                if isinstance(y, ProductionPredicterResult):
                    probPrint = f' [{y.probability:.2f}]' if logProb else ''
                    logger.info(f"Predicton for the past {chunkTime}sec: {y.label}{probPrint} | delay: {delayPrint}sec with prediction delay of {y.predictionDelay}sec")
                elif isinstance(y, EvaluationPredicterResult):
                    printRes_received = f'{y.forReceived.label} [{y.forReceived.probability:.2f}]' if logProb else f'{y.forReceived.label}'
                    printRes_played = f'{y.forPlayed.label} [{y.forPlayed.probability:.2f}]' if logProb else f'{y.forPlayed.label}'
                    printRes_gt = f'{y.groundTruth.label} [{y.groundTruth.probability:.2f}]' if logProb else f'{y.groundTruth.label}'
                    logger.info(f"Predicton of the past {chunkTime}sec: {printRes_received} received, {printRes_played} played, {printRes_gt} ground-truth | delay: {receiver.getDelay(seperateDelayLog)}sec, predDelay {y.predictionDelay}sec")
    except KeyboardInterrupt:
        logger.warning("Caught KeyboardInterrupt")
        logger.info('Stopped gracefully')
    finally:
        picke_lock = threading.Lock()
        picke_lock.acquire()
        receiver.closeReceiver()
        if saveRecords:
            utils.saveReceiverBuffer(receiverBuffer=receiver.buffer, path='./records/')
        picke_lock.release()

def nonBlocking_delayPrint(delayTime: float, logger: logging.Logger):
    time.sleep(delayTime)
    logger.warning(f"Message printed after delay of {delayTime}sec")

# def logger_thread(q):
#     while True:
#         record = q.get()
#         if record is None:
#             break
#         logger = logging.getLogger(record.name)
#         logger.handle(record)
#         time.sleep(0.001)

def main():
    fire.Fire(execute)

if __name__ == '__main__':
    main()