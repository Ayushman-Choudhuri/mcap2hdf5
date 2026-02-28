"""CLI entry point for mcap2hdf5."""

from pathlib import Path
from typing import Optional

import typer
from mcap.reader import make_reader
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="mcap2hdf5",
    help="Convert ROS2 MCAP recordings to synchronized HDF5 datasets.",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()

_CAMERA_IMAGE_TYPES = {"sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image"}
_CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"
_POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"


@app.callback()
def main(
    ctx: typer.Context,
    mcapPath: Optional[Path] = typer.Option(
        None,
        "--inspect",
        help="Inspect an MCAP file: list topics and show auto-detected sensor assignments.",
        metavar="MCAP_PATH",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    if mcapPath is None:
        typer.echo(ctx.get_help())
        return

    if not mcapPath.exists():
        console.print(f"[red]Error:[/red] File not found: {mcapPath}")
        raise typer.Exit(code=1)

    try:
        with open(mcapPath, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read MCAP file: {e}")
        raise typer.Exit(code=1)

    if summary is None:
        console.print("[yellow]Warning:[/yellow] MCAP file contains no summary block.")
        raise typer.Exit(code=1)

    if summary.statistics and summary.statistics.channel_message_counts:
        counts = summary.statistics.channel_message_counts
    else:
        counts = _countMessages(mcapPath)

    table = Table(title=f"[bold]{mcapPath.name}[/bold]", show_lines=True)
    table.add_column("Topic", style="cyan", no_wrap=True)
    table.add_column("Message Type", style="white")
    table.add_column("Count", style="green", justify="right")

    topicToSchema: dict[str, str] = {}
    for channelId, channel in sorted(summary.channels.items()):
        schema = summary.schemas.get(channel.schema_id)
        schemaName = schema.name if schema else "unknown"
        count = counts.get(channelId, 0)
        table.add_row(channel.topic, schemaName, str(count))
        topicToSchema[channel.topic] = schemaName

    console.print(table)

    console.print("\n[bold]Auto-detection:[/bold]")
    cameraImage = _detectFirst(topicToSchema, _CAMERA_IMAGE_TYPES, "camera image")
    cameraInfo = _detectFirst(topicToSchema, {_CAMERA_INFO_TYPE}, "camera info")
    lidar = _detectFirst(topicToSchema, {_POINTCLOUD2_TYPE}, "lidar")
    _printDetection("Camera image", cameraImage)
    _printDetection("Camera info ", cameraInfo)
    _printDetection("LiDAR       ", lidar)


def _countMessages(mcapPath: Path) -> dict[int, int]:
    """Count messages per channel by iterating raw records (no payload decoding)."""
    counts: dict[int, int] = {}
    with open(mcapPath, "rb") as f:
        reader = make_reader(f)
        for _, channel, _ in reader.iter_messages():
            counts[channel.id] = counts.get(channel.id, 0) + 1
    return counts


def _detectFirst(
    topicToSchema: dict[str, str], targetTypes: set[str], label: str
) -> str | None:
    matches = [t for t, s in topicToSchema.items() if s in targetTypes]
    if not matches:
        return None
    if len(matches) > 1:
        console.print(f"  [yellow]Note:[/yellow] Multiple {label} topics — using first: {matches}")
    return matches[0]


def _printDetection(label: str, topic: str | None) -> None:
    if topic:
        console.print(f"  {label}: [green]{topic}[/green]")
    else:
        console.print(f"  {label}: [red]not found[/red]")
