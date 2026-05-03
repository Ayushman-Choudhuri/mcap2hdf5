from typing import NamedTuple

from mcap2hdf5.configs.messages import (
    CAMERA_IMAGE_MESSAGE_TYPES,
    CAMERA_INFO_MESSAGE_TYPES,
    POINTCLOUD2_MESSAGE_TYPES,
    TF_MESSAGE_TYPES,
)


class DetectedSensors(NamedTuple):
    cameraImage: str | None
    cameraInfo: str | None
    lidar: str | None
    tf: str | None
    tfStatic: str | None


def detectSensors(topicToSchema: dict[str, str]) -> DetectedSensors:
    """Heuristically assign topics to sensor modalities based on message type."""

    tf, tfStatic = detectTF(topicToSchema)
    return DetectedSensors(
        cameraImage=detectFirst(topicToSchema, CAMERA_IMAGE_MESSAGE_TYPES),
        cameraInfo=detectFirst(topicToSchema, CAMERA_INFO_MESSAGE_TYPES),
        lidar=detectFirst(topicToSchema, POINTCLOUD2_MESSAGE_TYPES),
        tf=tf,
        tfStatic=tfStatic,
    )


def detectFirst(topicToSchema: dict[str, str], targetTypes: set[str]) -> str | None:
    """Return the first topic whose schema is in ``targetTypes``, or ``None``."""

    matches = [topic for topic, schemaName in topicToSchema.items() if schemaName in targetTypes]
    return matches[0] if matches else None


def detectAll(topicToSchema: dict[str, str], targetTypes: set[str]) -> list[str]:
    """Return all topics whose schema is in ``targetTypes``."""

    return [topic for topic, schemaName in topicToSchema.items() if schemaName in targetTypes]


def detectTF(topicToSchema: dict[str, str]) -> tuple[str | None, str | None]:
    """Detect dynamic TF and static TF topics by schema type and topic name."""

    static: list[str] = []
    dynamic: list[str] = []
    for topic, schema in topicToSchema.items():
        if schema in TF_MESSAGE_TYPES:
            (static if "static" in topic.lower() else dynamic).append(topic)

    return (dynamic[0] if dynamic else None), (static[0] if static else None)
