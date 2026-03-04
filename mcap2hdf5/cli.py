from pathlib import Path
from typing import Optional

import typer

from mcap2hdf5.cli_utils import (
    console,
    detectSensors,
    inspectMcap,
    printAutoDetection,
    printTopicTable,
)
from mcap2hdf5.job_config import JobConfig

app = typer.Typer(
    name="mcap2hdf5",
    help="Convert ROS2 MCAP recordings to synchronized HDF5 datasets.",
    no_args_is_help=True,
    invoke_without_command=True,
)


@app.callback()
def main(
    context: typer.Context,
    inspectPath: Optional[Path] = typer.Option(
        None,
        "--inspect",
        help="Inspect an MCAP file: list topics and show auto-detected sensor assignments.",
        metavar="MCAP_PATH",
    ),
    configPath: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Auto-detect modalities from an MCAP file and write a YAML job config.",
        metavar="MCAP_PATH",
    ),
) -> None:
    """Entry point for the mcap2hdf5 CLI."""
    
    if context.invoked_subcommand is not None:
        return

    if inspectPath is not None:
        runInspect(inspectPath)
    elif configPath is not None:
        runConfig(configPath)
    else:
        typer.echo(context.get_help())


def runInspect(mcapPath: Path) -> None:
    """Print a topic table and auto-detected sensor assignments for an MCAP file."""

    topicToSchema, topicCounts = inspectMcap(mcapPath)
    printTopicTable(mcapPath, topicToSchema, topicCounts)
    printAutoDetection(detectSensors(topicToSchema))


def runConfig(mcapPath: Path) -> None:
    """Auto-detect sensor topics from an MCAP file and write a YAML job config."""

    topicToSchema, topicCounts = inspectMcap(mcapPath)
    printTopicTable(mcapPath, topicToSchema, topicCounts)
    detectedSensors = detectSensors(topicToSchema)
    printAutoDetection(detectedSensors)

    cameraImage, cameraInfo, lidar, tf, tfStatic = detectedSensors
    jobConfig = JobConfig.from_detection(
        mcapPath,
        cameraImage,
        cameraInfo,
        lidar,
        tf,
        tfStatic
    )
    outputPath = Path(f"{mcapPath.stem}_config.yaml")
    jobConfig.save(outputPath)
    console.print(f"\n[bold green]Config written to:[/bold green] {outputPath}")
