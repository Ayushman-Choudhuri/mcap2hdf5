from pathlib import Path

import typer
from mcap.reader import make_reader
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
from mcap2hdf5.utils.logger import logger


def inspectMcap(mcapPath: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Read MCAP summary metadata and return per-topic schema names and message counts."""

    try:
        with open(mcapPath, "rb") as mcapFile:
            reader = make_reader(mcapFile)
            with logger.status("[dim]Reading MCAP summary...[/dim]"):
                summary = reader.get_summary()
    except FileNotFoundError:
        logger.error(f"File not found: {mcapPath}")
        raise typer.Exit(code=1) from None
    except Exception as e:
        logger.error(f"Failed to read MCAP file: {e}")
        raise typer.Exit(code=1) from None

    if summary is None:
        logger.warning("MCAP file contains no summary block.")
        raise typer.Exit(code=1)

    if summary.statistics and summary.statistics.channel_message_counts:
        logger.info("Using message counts from MCAP statistics record.")
        channelCounts = summary.statistics.channel_message_counts
    else:
        logger.info("No statistics record found — scanning file for message counts.")
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
    fileSize = mcapPath.stat().st_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=logger.console,
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

    logger.console.print(table)


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
            logger.warning(f"Multiple {label} topics — using first: {matches}")

    dynamic_all = [
        t for t, s in topicToSchema.items()
        if s in TF_MESSAGE_TYPES and "static" not in t.lower()
    ]
    static_all = [
        t for t, s in topicToSchema.items()
        if s in TF_MESSAGE_TYPES and "static" in t.lower()
    ]
    if len(dynamic_all) > 1:
        logger.warning(f"Multiple TF topics — using first: {dynamic_all}")
    if len(static_all) > 1:
        logger.warning(f"Multiple TF static topics — using first: {static_all}")

    logger.console.print("\n[bold]Auto-detection:[/bold]")
    _printDetection("Camera image", cameraImage)
    _printDetection("Camera info ", cameraInfo)
    _printDetection("LiDAR       ", lidar)
    _printDetection("TF          ", tf)
    _printDetection("TF static   ", tfStatic)


def _printDetection(label: str, topic: str | None) -> None:
    if topic:
        logger.console.print(f"  {label}: [green]{topic}[/green]")
    else:
        logger.console.print(f"  {label}: [red]not found[/red]")
