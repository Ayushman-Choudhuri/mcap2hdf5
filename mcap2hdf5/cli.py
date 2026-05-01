from pathlib import Path

import typer

from mcap2hdf5.pipeline import runPipeline
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


@app.command()
def convert(
    path: Path = typer.Argument(
        ...,
        help="Path to a job config YAML or an MCAP file (topics auto-detected).",
        metavar="PATH",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HDF5 path. Overrides the value in the job config.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose pipeline logging.",
    ),
) -> None:
    """Convert an MCAP recording to a synchronized HDF5 dataset."""

    if path.suffix in (".yaml", ".yml"):
        try:
            jobConfig = JobConfig.load(path)
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to load config: {e}")
            raise typer.Exit(code=1) from None
        if output is not None:
            jobConfig.outputHdf5 = str(output)

    elif path.suffix == ".mcap":
        topicToSchema, _ = inspectMcap(path)
        detectedSensors = detectSensors(topicToSchema)
        printAutoDetection(detectedSensors)
        cameraImage, cameraInfo, lidar, tf, tfStatic = detectedSensors
        jobConfig = JobConfig.from_detection(path, cameraImage, cameraInfo, lidar, tf, tfStatic)
        jobConfig.outputHdf5 = str(output if output is not None else Path(path.stem + ".hdf5"))

    else:
        console.print(
            f"[red]Error:[/red] Unrecognised file type '{path.suffix}'."
            " Expected .mcap or .yaml/.yml"
        )
        raise typer.Exit(code=1)

    m = jobConfig.modalities
    if not m.camera.imageTopic:
        console.print("[red]Error:[/red] No camera image topic configured or detected.")
        raise typer.Exit(code=1)
    if not m.lidar.topic:
        console.print("[red]Error:[/red] No LiDAR topic configured or detected.")
        raise typer.Exit(code=1)
    if not m.tf.topic:
        console.print("[red]Error:[/red] No TF topic configured or detected.")
        raise typer.Exit(code=1)

    try:
        runPipeline(jobConfig, verbose)
    except Exception:
        raise typer.Exit(code=1) from None
