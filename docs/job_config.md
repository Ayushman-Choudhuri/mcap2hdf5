# Job Config Reference

A job config is a YAML file that tells `mcap2hdf5 convert` where to read from, where to write, and how to synchronize sensors. You can generate one automatically with `mcap2hdf5 init`, then edit it to suit your setup.

## Generating a config

```bash
mcap2hdf5 init data/raw/recording.mcap
```

This inspects the MCAP file, auto-detects camera, LiDAR, and TF topics, and writes `recording_config.yaml` in the current directory. Open it in an editor and adjust any topics or thresholds before converting.

## Full example

```yaml
source:
  mcap: data/raw/recording.mcap

output:
  hdf5: data/processed/recording.hdf5

modalities:
  camera:
    image_topic: /camera/image_raw/compressed
    info_topic: /camera/camera_info
    sync:
      enabled: true
      algorithm: nearest
      threshold_sec: 0.05

  lidar:
    topic: /lidar/points
    sync:
      enabled: true
      reference: true

  tf:
    topic: /tf
    static_topic: /tf_static
    sync:
      enabled: false

pipeline:
  max_chunk_gap: 0.15
  hdf5_write_batch_size: 100
  tf_cache_size: 100
```

## Section reference

### `source`

| Key | Type | Description |
|-----|------|-------------|
| `mcap` | string | Path to the input MCAP file. Relative paths are resolved from the working directory. |

### `output`

| Key | Type | Description |
|-----|------|-------------|
| `hdf5` | string | Path for the output HDF5 file. Parent directories are created automatically. |

### `modalities.camera`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `image_topic` | string | — | **Required.** Topic publishing camera frames. Supported message types: `sensor_msgs/msg/Image`, `sensor_msgs/msg/CompressedImage`, `foxglove.CompressedImage`. |
| `info_topic` | string | `null` | Topic publishing `sensor_msgs/msg/CameraInfo`. Used to write intrinsic calibration (K, D, R, P matrices) as HDF5 attributes. Optional but recommended. |
| `sync.enabled` | bool | `true` | Whether to synchronize camera frames to the LiDAR reference clock. Set to `false` only for debugging. |
| `sync.algorithm` | string | `nearest` | Matching strategy. Only `nearest` is currently supported. |
| `sync.threshold_sec` | float | `0.05` | Maximum time difference (seconds) between a camera frame and its matched LiDAR sweep. Frames outside this window are dropped. Lower values are stricter; raise it if your sensors have loose hardware synchronization. |

### `modalities.lidar`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `topic` | string | — | **Required.** Topic publishing `sensor_msgs/msg/PointCloud2` sweeps. |
| `sync.enabled` | bool | `true` | Must be `true`. LiDAR is always the reference sensor. |
| `sync.reference` | bool | `true` | Marks LiDAR as the synchronization clock. Each new LiDAR sweep triggers a synchronization attempt for all other modalities. Keep this `true`. |

### `modalities.tf`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `topic` | string | — | **Required.** Topic publishing `tf2_msgs/msg/TFMessage` (dynamic transforms, typically `/tf`). |
| `static_topic` | string | `null` | Topic publishing static transforms (typically `/tf_static`). These are written once to HDF5 file attributes. Optional but needed if your sensor extrinsics are broadcast on `/tf_static`. |
| `sync.enabled` | bool | `false` | TF messages are cached and interpolated rather than synchronized frame-by-frame. Leave this `false`. |

### `pipeline`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_chunk_gap` | float | `0.15` | Maximum time gap (seconds) between consecutive messages on the same topic before the current batch is flushed to HDF5 as a new chunk. Increase for recordings with natural pauses; decrease to split more aggressively. |
| `hdf5_write_batch_size` | int | `100` | Number of synchronized samples accumulated in memory before a batch write to disk. Higher values reduce I/O overhead but increase memory usage. |
| `tf_cache_size` | int | `100` | Number of TF entries kept per transform pair for interpolation. Increase if your TF publish rate is low relative to the recording duration. |

## Supported message types

| Modality | ROS2 message type |
|----------|-------------------|
| Camera image | `sensor_msgs/msg/Image` |
| Camera image (compressed) | `sensor_msgs/msg/CompressedImage` |
| Camera image (Foxglove) | `foxglove.CompressedImage` |
| Camera calibration | `sensor_msgs/msg/CameraInfo` |
| LiDAR | `sensor_msgs/msg/PointCloud2` |
| Dynamic transforms | `tf2_msgs/msg/TFMessage` |
| Static transforms | `tf2_msgs/msg/TFMessage` |

## Choosing `threshold_sec`

The synchronization threshold is the single most important tuning parameter. A value too small drops frames; too large produces misaligned pairs.

- **Hardware-synced rigs** (camera trigger locked to LiDAR): `0.01`–`0.02` s is usually safe.
- **Software-synced rigs** (independent clocks): `0.05` s is a reasonable starting point.
- **High-motion sequences**: tighten the threshold to reduce motion blur between camera and LiDAR.

Run `mcap2hdf5 inspect` first and check the message frequencies printed in the topic table. If the camera publishes at 10 Hz and LiDAR at 10 Hz, a threshold of `0.05` s gives ±half-frame tolerance.

## Common adjustments

**Single-camera, no calibration file**

```yaml
modalities:
  camera:
    image_topic: /camera/image_raw
    info_topic: null   # omit calibration attributes from HDF5
```

**No `/tf_static` in the recording**

```yaml
modalities:
  tf:
    topic: /tf
    static_topic: null
```

**Recording with long stationary periods**

Increase `max_chunk_gap` so pauses don't split the data into many small chunks:

```yaml
pipeline:
  max_chunk_gap: 2.0
```

**Override the output path without editing the config**

Edit the `output.hdf5` field directly in the YAML, or duplicate the config file for each output path variant.
