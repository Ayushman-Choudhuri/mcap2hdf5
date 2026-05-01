import logging
from pathlib import Path

from mcap2hdf5.configs.names import CHUNK_ID
from mcap2hdf5.hdf5_writer import HDF5Writer
from mcap2hdf5.reader import MCAPSource
from mcap2hdf5.synchronizer import SensorDataSynchronizer
from mcap2hdf5.utils.job_config import JobConfig


def runPipeline(jobConfig: JobConfig, verbose: bool = False) -> None:
    """Execute the MCAPSource → SensorDataSynchronizer → HDF5Writer pipeline."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    console = Console()

    mcapPath = Path(jobConfig.sourceMcap)
    hdf5Path = Path(jobConfig.outputHdf5)
    hdf5Path.parent.mkdir(parents=True, exist_ok=True)

    m = jobConfig.modalities
    p = jobConfig.pipeline

    topics = [t for t in [
        m.camera.imageTopic,
        m.camera.infoTopic,
        m.lidar.topic,
        m.tf.topic,
        m.tf.staticTopic,
    ] if t is not None]

    source = MCAPSource(
        mcapPath,
        topics=topics,
        cameraInfoTopic=m.camera.infoTopic,
        tfStaticTopic=m.tf.staticTopic,
    )
    synchronizer = SensorDataSynchronizer(
        syncThreshold=m.camera.sync.thresholdSec,
        maxGap=p.maxChunkGap,
        cameraImageTopic=m.camera.imageTopic,
        lidarTopic=m.lidar.topic,
        tfTopic=m.tf.topic,
    )
    writer = HDF5Writer(hdf5Path)

    sampleBatch = []
    chunkId = 0
    totalSamples = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Converting...", total=None)

            for streamMessage in source.streamMessages():
                flushEventTriggered = False

                for sample in synchronizer.processMessage(streamMessage):
                    flushEventTriggered = True
                    sample[CHUNK_ID] = chunkId
                    sampleBatch.append(sample)

                    if len(sampleBatch) >= p.hdf5WriteBatchSize:
                        writer.writeBatch(sampleBatch)
                        totalSamples += len(sampleBatch)
                        progress.update(
                            task,
                            description=(
                                f"Converting... {totalSamples} samples, {chunkId} chunks"
                            ),
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

            progress.update(
                task,
                description=(
                    f"[green]Done[/green] — {totalSamples} samples, {chunkId + 1} chunks"
                ),
            )

        writer.finalize(
            cameraMetadata=source.getCameraMetadata(),
            staticTransforms=source.getStaticTransforms(),
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] Conversion failed: {e}")
        raise

    console.print(f"\n[bold green]Output:[/bold green] {hdf5Path}")
