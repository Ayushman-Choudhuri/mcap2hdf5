import logging
from pipeline.config import (
    CHUNKS_FILE_PATH,
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
            for sample in synchronizer.processMessage(streamMessage):
                sampleBatch.append(sample)

                if len(sampleBatch) >= HDF5_WRITE_BATCH_SIZE:
                    writer.writeBatch(sampleBatch, chunkId)
                    totalSamples += len(sampleBatch)
                    logger.info(f"Processed {totalSamples} samples...")
                    sampleBatch = []
                    chunkId += 1


        finalSamples = list(synchronizer.flushSamples())
        if finalSamples:
            sampleBatch.extend(finalSamples)

        if sampleBatch:
            writer.writeBatch(sampleBatch, chunkId)
            totalSamples += len(sampleBatch)

        logger.info("Finalizing HDF5 file with metadata...")
        writer.finalize(
            cameraMetadata=source.getCameraMetadata(),
            staticTransforms=synchronizer.staticTransforms
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    
    logger.info("Conversion complete!")

if __name__ == "__main__":
    main()