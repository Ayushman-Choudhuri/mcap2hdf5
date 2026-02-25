from dataclasses import dataclass
from typing import (
    Any,
    Dict,
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
    transforms: Dict[str, Any]
