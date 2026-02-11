import logging
from pipeline.config import (
    CHUNKS_FILE_PATH,
    CHUNK_ID,
    HDF5_WRITE_BATCH_SIZE,
    MAX_CHUNK_GAP,
    MCAP_FILE_PATH,
    SENSOR_SYNC_THRESHOLD,
)
from pipeline.hdf5_writer import HDF5Writer
from pipeline.reader import MCAPSource
from pipeline.synchronizer import SensorSynchronizer

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Pipeline")

    source = MCAPSource(MCAP_FILE_PATH)
    synchronizer = SensorSynchronizer(SENSOR_SYNC_THRESHOLD, MAX_CHUNK_GAP)
    writer = HDF5Writer(CHUNKS_FILE_PATH)

    logger.info("Starting conversion pipeline...")
    
    sampleBatch = []
    chunkId = 0
    totalSamples = 0

    try: 
        for streamMessage in source.streamMessages():
            flushEventTriggered = False

            for sample in synchronizer.processMessage(streamMessage):
                flushEventTriggered = True
                sample[CHUNK_ID] = chunkId
                sampleBatch.append(sample)

                if len(sampleBatch) >= HDF5_WRITE_BATCH_SIZE:
                    writer.writeBatch(sampleBatch)
                    totalSamples += len(sampleBatch)
                    logger.info(f"Written batch of {len(sampleBatch)} samples to HDF5.")
                    sampleBatch = []
            
            if flushEventTriggered:
                logger.info(f"Flush event triggered for chunk {chunkId} at timestamp {streamMessage.timestamp}.")
                chunkId += 1
        
        """ Handling of residual samples """

        for sample in synchronizer.flushSamples():
            sample[CHUNK_ID] = chunkId
            sampleBatch.append(sample)

        if sampleBatch:
            writer.writeBatch(sampleBatch)
            totalSamples += len(sampleBatch)
            logger.info(f"Written residual batch of {len(sampleBatch)} samples to HDF5.")
        
        writer.finalize(
            cameraMetadata=source.getCameraMetadata(),
            staticTransforms=source.getStaticTransforms(),
        )

    except Exception as e:
        logger.error(f"An error occurred during dataset generation: {e}")

    logger.info(f"Dataset generation completed with {totalSamples} samples across {chunkId+1} chunks.")

if __name__ == "__main__":
    main()