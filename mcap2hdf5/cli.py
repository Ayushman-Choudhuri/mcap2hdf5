from pathlib import Path

import typer

from mcap2hdf5.pipeline import runPipeline
from mcap2hdf5.utils.cli_utils import (
    inspectMcap,
    printAutoDetection,
    printTopicTable,
)
from mcap2hdf5.utils.detect import detectSensors
from mcap2hdf5.utils.job_config import JobConfig
from mcap2hdf5.utils.logger import logger

app = typer.Typer(
    name="mcap2hdf5",
    help="Convert ROS2 MCAP recordings to synchronized HDF5 datasets.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def inspect(
    path: Path = typer.Argument(
        ...,
        help="Path to the MCAP file.",
        metavar="MCAP_PATH",
    ),
) -> None:
    """List topics and show auto-detected sensor assignments for an MCAP file."""

    topicToSchema, topicCounts = inspectMcap(path)
    printTopicTable(path, topicToSchema, topicCounts)
    printAutoDetection(topicToSchema, detectSensors(topicToSchema))


@app.command()
def init(
    path: Path = typer.Argument(
        ...,
        help="Path to the MCAP file.",
        metavar="MCAP_PATH",
    ),
) -> None:
    """Auto-detect sensor topics from an MCAP file and write a YAML job config."""

    topicToSchema, topicCounts = inspectMcap(path)
    printTopicTable(path, topicToSchema, topicCounts)
    detectedSensors = detectSensors(topicToSchema)
    printAutoDetection(topicToSchema, detectedSensors)

    cameraImage, cameraInfo, lidar, tf, tfStatic = detectedSensors
    jobConfig = JobConfig.from_detection(
        path,
        cameraImage,
        cameraInfo,
        lidar,
        tf,
        tfStatic,
    )
    outputPath = Path(f"{path.stem}_config.yaml")
    jobConfig.save(outputPath)
    logger.console.print(f"\n[bold green]Config written to:[/bold green] {outputPath}")


@app.command()
def convert(
    path: Path = typer.Argument(
        ...,
        help="Path to a job config YAML file.",
        metavar="JOB_CONFIG_PATH",
    ),
) -> None:
    """Convert an MCAP recording to a synchronized HDF5 dataset."""

    if path.suffix not in (".yaml", ".yml"):
        logger.error(
            f"Expected a .yaml or .yml job config, got '{path.suffix}'. "
            "Run `mcap2hdf5 init <file.mcap>` to generate one automatically or write manually. "
            "See the documentation for details."
        )
        raise typer.Exit(code=1)

    try:
        jobConfig = JobConfig.load(path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise typer.Exit(code=1) from None

    try:
        runPipeline(jobConfig)
    except Exception:
        raise typer.Exit(code=1) from None
