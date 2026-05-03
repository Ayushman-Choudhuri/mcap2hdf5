from pathlib import Path

from rich.progress import (
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from mcap2hdf5.configs.names import CHUNK_ID
from mcap2hdf5.hdf5_writer import HDF5Writer
from mcap2hdf5.reader import MCAPSource
from mcap2hdf5.synchronizer import SensorDataSynchronizer
from mcap2hdf5.utils.job_config import JobConfig
from mcap2hdf5.utils.logger import logger


def _buildComponents(
    jobConfig: JobConfig, hdf5Path: Path
) -> tuple[MCAPSource, SensorDataSynchronizer, HDF5Writer]:
    """Construct and return the three pipeline stage objects from a job config."""
    modalities = jobConfig.modalities
    pipeline = jobConfig.pipeline
    mcapPath = Path(jobConfig.sourceMcap)

    topics = [topic for topic in [
        modalities.camera.imageTopic,
        modalities.camera.infoTopic,
        modalities.lidar.topic,
        modalities.tf.topic,
        modalities.tf.staticTopic,
    ] if topic is not None]

    source = MCAPSource(
        mcapPath,
        topics=topics,
        cameraInfoTopic=modalities.camera.infoTopic,
        tfStaticTopic=modalities.tf.staticTopic,
    )

    synchronizer = SensorDataSynchronizer(
        syncThreshold=modalities.camera.sync.thresholdSec,
        maxGap=pipeline.maxChunkGap,
        cameraImageTopic=modalities.camera.imageTopic,
        lidarTopic=modalities.lidar.topic,
        tfTopic=modalities.tf.topic,
    )

    writer = HDF5Writer(hdf5Path)

    return source, synchronizer, writer


def _runConversionLoop(
    source: MCAPSource,
    synchronizer: SensorDataSynchronizer,
    writer: HDF5Writer,
    batchSize: int,
    progress: Progress,
    task: TaskID,
) -> tuple[int, int]:
    """Stream messages, synchronize, and write batches. Returns (totalSamples, chunkId)."""
    sampleBatch = []
    chunkId = 0
    totalSamples = 0

    for streamMessage in source.streamMessages():
        flushEventTriggered = False

        for sample in synchronizer.processMessage(streamMessage):
            flushEventTriggered = True
            sample[CHUNK_ID] = chunkId
            sampleBatch.append(sample)

            if len(sampleBatch) >= batchSize:
                writer.writeBatch(sampleBatch)
                totalSamples += len(sampleBatch)
                progress.update(
                    task,
                    description=f"Converting... {totalSamples} samples, {chunkId} chunks",
                )
                sampleBatch = []

        if flushEventTriggered:
            chunkId += 1

    for sample in synchronizer.flushSamples():
        sample[CHUNK_ID] = chunkId
        sampleBatch.append(sample)

    if sampleBatch:
        writer.writeBatch(sampleBatch)
        totalSamples += len(sampleBatch)

    return totalSamples, chunkId


def runPipeline(jobConfig: JobConfig) -> None:
    """Execute the MCAPSource → SensorDataSynchronizer → HDF5Writer conversion pipeline."""
    hdf5Path = Path(jobConfig.outputHdf5)
    hdf5Path.parent.mkdir(parents=True, exist_ok=True)
    source, synchronizer, writer = _buildComponents(jobConfig, hdf5Path)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=logger.console,
        ) as progress:
            task = progress.add_task("Converting...", total=None)

            totalSamples, chunkId = _runConversionLoop(
                source,
                synchronizer,
                writer,
                jobConfig.pipeline.hdf5WriteBatchSize,
                progress,
                task,
            )

            progress.update(
                task,
                description=f"[green]Done[/green] — {totalSamples} samples, {chunkId + 1} chunks",
            )

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

    try:
        writer.finalize(
            cameraMetadata=source.getCameraMetadata(),
            staticTransforms=source.getStaticTransforms(),
        )
    except Exception as e:
        logger.error(f"Failed to finalize HDF5 output: {e}")
        raise

    logger.info(f"Output: {hdf5Path}")
