from pathlib import Path

import typer
from mcap.reader import make_reader
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

from mcap2hdf5.configs.messages import (
    CAMERA_IMAGE_MESSAGE_TYPES,
    CAMERA_INFO_MESSAGE_TYPES,
    POINTCLOUD2_MESSAGE_TYPES,
    TF_MESSAGE_TYPES,
)
from mcap2hdf5.utils.detect import DetectedSensors, detectAll

console = Console()


def inspectMcap(mcapPath: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Read MCAP summary metadata and return per-topic schema names and message counts."""

    try:
        with open(mcapPath, "rb") as mcapFile:
            reader = make_reader(mcapFile)
            with console.status("[dim]Reading MCAP summary...[/dim]"):
                summary = reader.get_summary()
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {mcapPath}")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read MCAP file: {e}")
        raise typer.Exit(code=1) from None

    if summary is None:
        console.print("[yellow]Warning:[/yellow] MCAP file contains no summary block.")
        raise typer.Exit(code=1)

    if summary.statistics and summary.statistics.channel_message_counts:
        console.print("[dim]Using message counts from MCAP statistics record.[/dim]")
        channelCounts = summary.statistics.channel_message_counts
    else:
        console.print("[dim]No statistics record found — scanning file for message counts.[/dim]")
        channelCounts = countMessagesByChannel(mcapPath)

    topicToSchema: dict[str, str] = {}
    topicCounts: dict[str, int] = {}
    for channelId, channel in sorted(summary.channels.items()):
        schema = summary.schemas.get(channel.schema_id)
        topicToSchema[channel.topic] = schema.name if schema else "unknown"
        topicCounts[channel.topic] = channelCounts.get(channelId, 0)

    print(topicToSchema)
    print(topicCounts)

    return topicToSchema, topicCounts


def countMessagesByChannel(mcapPath: Path) -> dict[int, int]:
    """Scan every message in an MCAP file and count messages per channel ID."""

    counts: dict[int, int] = {}
    fileSize = mcapPath.stat().st_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning messages (no summary record found)...", total=fileSize)
        with open(mcapPath, "rb") as f:
            reader = make_reader(f)
            for msgIdx, (_, channel, _) in enumerate(reader.iter_messages()):
                counts[channel.id] = counts.get(channel.id, 0) + 1
                if msgIdx % 200 == 0:
                    progress.update(task, completed=f.tell())
        progress.update(task, completed=fileSize)

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


def printAutoDetection(topicToSchema: dict[str, str], detected: DetectedSensors) -> None:
    """Print auto-detected sensor assignments, warning on ambiguous multi-topic matches."""

    cameraImage, cameraInfo, lidar, tf, tfStatic = detected

    for label, types in (
        ("camera image", CAMERA_IMAGE_MESSAGE_TYPES),
        ("camera info", CAMERA_INFO_MESSAGE_TYPES),
        ("lidar", POINTCLOUD2_MESSAGE_TYPES),
    ):
        matches = detectAll(topicToSchema, types)
        if len(matches) > 1:
            console.print(
                f"  [yellow]Note:[/yellow] Multiple {label} topics — using first: {matches}"
            )

    dynamic_all = [
        t for t, s in topicToSchema.items()
        if s in TF_MESSAGE_TYPES and "static" not in t.lower()
    ]
    static_all = [
        t for t, s in topicToSchema.items()
        if s in TF_MESSAGE_TYPES and "static" in t.lower()
    ]
    if len(dynamic_all) > 1:
        console.print(
            f"  [yellow]Note:[/yellow] Multiple TF topics — using first: {dynamic_all}"
        )
    if len(static_all) > 1:
        console.print(
            f"  [yellow]Note:[/yellow] Multiple TF static topics — using first: {static_all}"
        )

    console.print("\n[bold]Auto-detection:[/bold]")
    _printDetection("Camera image", cameraImage)
    _printDetection("Camera info ", cameraInfo)
    _printDetection("LiDAR       ", lidar)
    _printDetection("TF          ", tf)
    _printDetection("TF static   ", tfStatic)


def _printDetection(label: str, topic: str | None) -> None:
    if topic:
        console.print(f"  {label}: [green]{topic}[/green]")
    else:
        console.print(f"  {label}: [red]not found[/red]")
