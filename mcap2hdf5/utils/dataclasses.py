from dataclasses import dataclass
from typing import (
    Any,
)


@dataclass
class StreamMessage:
    topic: str
    msg: Any
    timestamp: float


@dataclass
class SyncGroup:
    timestamp: float
    lidar: Any
    camera: Any
    transforms: dict[str, Any]
