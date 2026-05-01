from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mcap2hdf5.configs.hdf5 import HDF5_WRITE_BATCH_SIZE
from mcap2hdf5.configs.pipeline import (
    MAX_CHUNK_GAP,
    SENSOR_SYNC_THRESHOLD,
    TF_CACHE_SIZE,
)


@dataclass
class SyncConfig:
    enabled: bool
    algorithm: str = "nearest"
    thresholdSec: float = SENSOR_SYNC_THRESHOLD
    reference: bool = False

    def to_dict(self) -> dict:
        d: dict = {"enabled": self.enabled}
        if self.enabled:
            d["algorithm"] = self.algorithm
            d["threshold_sec"] = self.thresholdSec
        if self.reference:
            d["reference"] = self.reference
        return d


@dataclass
class CameraConfig:
    imageTopic: str | None
    infoTopic: str | None
    sync: SyncConfig = field(default_factory=lambda: SyncConfig(enabled=True))

    def to_dict(self) -> dict:
        return {
            "image_topic": self.imageTopic,
            "info_topic": self.infoTopic,
            "sync": self.sync.to_dict(),
        }


@dataclass
class LidarConfig:
    topic: str | None
    sync: SyncConfig = field(default_factory=lambda: SyncConfig(enabled=True, reference=True))

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "sync": self.sync.to_dict(),
        }


@dataclass
class TFConfig:
    topic: str | None = None
    staticTopic: str | None = None
    sync: SyncConfig = field(default_factory=lambda: SyncConfig(enabled=False))

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "static_topic": self.staticTopic,
            "sync": self.sync.to_dict(),
        }


@dataclass
class ModalitiesConfig:
    camera: CameraConfig
    lidar: LidarConfig
    tf: TFConfig

    def to_dict(self) -> dict:
        return {
            "camera": self.camera.to_dict(),
            "lidar": self.lidar.to_dict(),
            "tf": self.tf.to_dict(),
        }


@dataclass
class PipelineConfig:
    maxChunkGap: float = MAX_CHUNK_GAP
    hdf5WriteBatchSize: int = HDF5_WRITE_BATCH_SIZE
    tfCacheSize: int = TF_CACHE_SIZE

    def to_dict(self) -> dict:
        return {
            "max_chunk_gap": self.maxChunkGap,
            "hdf5_write_batch_size": self.hdf5WriteBatchSize,
            "tf_cache_size": self.tfCacheSize,
        }


@dataclass
class JobConfig:
    """Runtime configuration for a single mcap2hdf5 conversion job."""

    sourceMcap: str
    outputHdf5: str
    modalities: ModalitiesConfig
    pipeline: PipelineConfig

    @classmethod
    def from_detection(
        cls,
        mcapPath: Path,
        cameraImage: str | None,
        cameraInfo: str | None,
        lidar: str | None,
        tf: str | None = None,
        tfStatic: str | None = None,
    ) -> JobConfig:
        """Build a JobConfig from MCAP auto-detection results."""
        outputHdf5 = f"data/processed/{mcapPath.stem}.hdf5"
        return cls(
            sourceMcap=str(mcapPath),
            outputHdf5=outputHdf5,
            modalities=ModalitiesConfig(
                camera=CameraConfig(imageTopic=cameraImage, infoTopic=cameraInfo),
                lidar=LidarConfig(topic=lidar),
                tf=TFConfig(topic=tf, staticTopic=tfStatic),
            ),
            pipeline=PipelineConfig(),
        )

    def to_dict(self) -> dict:
        return {
            "source": {"mcap": self.sourceMcap},
            "output": {"hdf5": self.outputHdf5},
            "modalities": self.modalities.to_dict(),
            "pipeline": self.pipeline.to_dict(),
        }

    @classmethod
    def load(cls, path: Path) -> JobConfig:
        """Deserialize a JobConfig from a YAML file produced by save()."""
        with open(path) as f:
            data = yaml.safe_load(f)

        cam = data["modalities"]["camera"]
        cam_sync = cam.get("sync", {})
        lidar = data["modalities"]["lidar"]
        lidar_sync = lidar.get("sync", {})
        tf = data["modalities"]["tf"]
        tf_sync = tf.get("sync", {})
        pipeline = data.get("pipeline", {})

        return cls(
            sourceMcap=data["source"]["mcap"],
            outputHdf5=data["output"]["hdf5"],
            modalities=ModalitiesConfig(
                camera=CameraConfig(
                    imageTopic=cam.get("image_topic"),
                    infoTopic=cam.get("info_topic"),
                    sync=SyncConfig(
                        enabled=cam_sync.get("enabled", True),
                        algorithm=cam_sync.get("algorithm", "nearest"),
                        thresholdSec=cam_sync.get("threshold_sec", SENSOR_SYNC_THRESHOLD),
                    ),
                ),
                lidar=LidarConfig(
                    topic=lidar.get("topic"),
                    sync=SyncConfig(
                        enabled=lidar_sync.get("enabled", True),
                        reference=lidar_sync.get("reference", True),
                    ),
                ),
                tf=TFConfig(
                    topic=tf.get("topic"),
                    staticTopic=tf.get("static_topic"),
                    sync=SyncConfig(enabled=tf_sync.get("enabled", False)),
                ),
            ),
            pipeline=PipelineConfig(
                maxChunkGap=pipeline.get("max_chunk_gap", MAX_CHUNK_GAP),
                hdf5WriteBatchSize=pipeline.get("hdf5_write_batch_size", HDF5_WRITE_BATCH_SIZE),
                tfCacheSize=pipeline.get("tf_cache_size", TF_CACHE_SIZE),
            ),
        )

    def save(self, path: Path) -> None:
        """Serialize to YAML at the given path."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
