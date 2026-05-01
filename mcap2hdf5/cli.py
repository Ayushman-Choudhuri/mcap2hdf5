from pathlib import Path

import typer

from mcap2hdf5.utils.cli_utils import (
    console,
    detectSensors,
    inspectMcap,
    printAutoDetection,
    printTopicTable,
)
from mcap2hdf5.utils.job_config import JobConfig

app = typer.Typer(
    name="mcap2hdf5",
    help="Convert ROS2 MCAP recordings to synchronized HDF5 datasets.",
    no_args_is_help=True,
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
    printAutoDetection(detectSensors(topicToSchema))


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
    printAutoDetection(detectedSensors)

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
    console.print(f"\n[bold green]Config written to:[/bold green] {outputPath}")
