from pathlib import Path

import typer
from mcap.reader import make_reader
from rich.console import Console
from rich.table import Table

from mcap2hdf5.configs.messages import (
    CAMERA_IMAGE_MESSAGE_TYPES,
    CAMERA_INFO_MESSAGE_TYPES,
    POINTCLOUD2_MESSAGE_TYPES,
    TF_MESSAGE_TYPES,
)
from mcap2hdf5.dataclasses import DetectedSensors

console = Console()


def inspectMcap(mcapPath: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Read MCAP summary metadata and return per-topic schema names and message counts."""
    
    try:
        with open(mcapPath, "rb") as mcapFile:
            reader = make_reader(mcapFile)
            summary = reader.get_summary()
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {mcapPath}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read MCAP file: {e}")
        raise typer.Exit(code=1)

    if summary is None:
        console.print("[yellow]Warning:[/yellow] MCAP file contains no summary block.")
        raise typer.Exit(code=1)

    if summary.statistics and summary.statistics.channel_message_counts:
        channelCounts = summary.statistics.channel_message_counts
    else:
        channelCounts = countMessagesByChannel(mcapPath)

    topicToSchema: dict[str, str] = {}
    topicCounts: dict[str, int] = {}
    for channelId, channel in sorted(summary.channels.items()):
        schema = summary.schemas.get(channel.schema_id)
        topicToSchema[channel.topic] = schema.name if schema else "unknown"
        topicCounts[channel.topic] = channelCounts.get(channelId, 0)

    return topicToSchema, topicCounts


def countMessagesByChannel(mcapPath: Path) -> dict[int, int]:
    """Scan every message in an MCAP file and count messages per channel ID."""
    
    counts: dict[int, int] = {}
    with open(mcapPath, "rb") as f:
        reader = make_reader(f)
        for _, channel, _ in reader.iter_messages():
            counts[channel.id] = counts.get(channel.id, 0) + 1
    return counts


def printTopicTable(
    mcapPath: Path,
    topicToSchema: dict[str, str],
    topicCounts: dict[str, int],
) -> None:
    """Render a Rich table of topics, message types, and counts to the console."""
    
    table = Table(title=f"[bold]{mcapPath.name}[/bold]", show_lines=True)
    table.add_column("Topic", style="cyan", no_wrap=True)
    table.add_column("Message Type", style="white")
    table.add_column("Count", style="green", justify="right")

    for topic, schema in topicToSchema.items():
        table.add_row(topic, schema, str(topicCounts.get(topic, 0)))

    console.print(table)


def detectSensors(topicToSchema: dict[str, str]) -> DetectedSensors:
    """Heuristically assign topics to sensor modalities based on message type."""
    
    tf, tfStatic = detectTF(topicToSchema)
    return DetectedSensors(
        cameraImage=detectFirst(topicToSchema, CAMERA_IMAGE_MESSAGE_TYPES, "camera image"),
        cameraInfo=detectFirst(topicToSchema, CAMERA_INFO_MESSAGE_TYPES, "camera info"),
        lidar=detectFirst(topicToSchema, POINTCLOUD2_MESSAGE_TYPES, "lidar"),
        tf=tf,
        tfStatic=tfStatic,
    )


def detectFirst(
    topicToSchema: dict[str, str],
    targetTypes: set[str],
    label: str,
) -> str | None:
    """Return the first topic whose schema is in ``targetTypes``, or ``None``."""
    
    matches = [t for t, s in topicToSchema.items() if s in targetTypes]
    if not matches:
        return None
    if len(matches) > 1:
        console.print(f"  [yellow]Note:[/yellow] Multiple {label} topics — using first: {matches}")
    return matches[0]


def detectTF(topicToSchema: dict[str, str]) -> tuple[str | None, str | None]:
    """Detect dynamic TF and static TF topics by schema type and topic name."""
    
    static: list[str] = []
    dynamic: list[str] = []
    for topic, schema in topicToSchema.items():
        if schema in TF_MESSAGE_TYPES:
            (static if "static" in topic.lower() else dynamic).append(topic)

    if len(dynamic) > 1:
        console.print(f"  [yellow]Note:[/yellow] Multiple TF topics — using first: {dynamic}")
    if len(static) > 1:
        console.print(f"  [yellow]Note:[/yellow] Multiple TF static topics — using first: {static}")

    return (dynamic[0] if dynamic else None), (static[0] if static else None)


def printAutoDetection(detected: DetectedSensors) -> None:
    """Print the auto-detected sensor topic assignments to the console."""
    
    cameraImage, cameraInfo, lidar, tf, tfStatic = detected
    console.print("\n[bold]Auto-detection:[/bold]")
    printDetection("Camera image", cameraImage)
    printDetection("Camera info ", cameraInfo)
    printDetection("LiDAR       ", lidar)
    printDetection("TF          ", tf)
    printDetection("TF static   ", tfStatic)


def printDetection(label: str, topic: str | None) -> None:
    """Print a single sensor detection result, coloured green (found) or red (missing)."""
    
    if topic:
        console.print(f"  {label}: [green]{topic}[/green]")
    else:
        console.print(f"  {label}: [red]not found[/red]")
