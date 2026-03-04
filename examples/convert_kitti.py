import logging

from mcap2hdf5.configs.hdf5 import HDF5_WRITE_BATCH_SIZE
from mcap2hdf5.configs.pipeline import (
    MAX_CHUNK_GAP,
    SENSOR_SYNC_THRESHOLD,
)
from mcap2hdf5.hdf5_writer import CHUNK_ID, HDF5Writer
from mcap2hdf5.reader import MCAPSource
from mcap2hdf5.synchronizer import SensorDataSynchronizer

CAMERA_IMAGE_TOPIC = "/sensor/camera/left/image_raw/compressed"
CAMERA_INFO_TOPIC = "/sensor/camera/left/camera_info"
LIDAR_TOPIC = "/sensor/lidar/front/points"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Pipeline")

    source = MCAPSource(
        "data/raw/kitti.mcap",
        cameraInfoTopic=CAMERA_INFO_TOPIC,
        tfStaticTopic=TF_STATIC_TOPIC,
    )
    synchronizer = SensorDataSynchronizer(
        SENSOR_SYNC_THRESHOLD,
        MAX_CHUNK_GAP,
        cameraImageTopic=CAMERA_IMAGE_TOPIC,
        lidarTopic=LIDAR_TOPIC,
        tfTopic=TF_TOPIC,
    )
    writer = HDF5Writer("data/processed/chunks.hdf5")

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
                logger.info(
                    f"Flush event triggered for chunk {chunkId}"
                    f" at timestamp {streamMessage.timestamp}."
                )
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

    logger.info(
        f"Dataset generation completed with {totalSamples} samples across {chunkId+1} chunks."
    )

if __name__ == "__main__":
    main()