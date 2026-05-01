<h1 align="center">mcap2hdf5</h1>

<p align="center">
  <a href="https://github.com/Ayushman-Choudhuri/mcap2hdf5/actions/workflows/build.yml"><img src="https://github.com/Ayushman-Choudhuri/mcap2hdf5/actions/workflows/build.yml/badge.svg" alt="build"/></a>
  <a href="https://github.com/Ayushman-Choudhuri/mcap2hdf5/actions/workflows/style.yml"><img src="https://github.com/Ayushman-Choudhuri/mcap2hdf5/actions/workflows/style.yml/badge.svg" alt="style"/></a>
</p>

<p align="center">
  Convert ROS2 MCAP recordings to temporally-synchronized, randomly-accessible HDF5 datasets for ML training.
</p>

> **Status: active development — v0.1.0 pre-release. APIs and schema may change.**

---

## 1. What It Does

Reads an MCAP file, synchronizes LiDAR and camera frames by timestamp, interpolates coordinate transforms, and writes a compressed HDF5 dataset that any DataLoader can index in O(1).

The pipeline: `MCAPSource → SensorDataSynchronizer → HDF5Writer`

- LiDAR frames are used as the sync reference. Each frame is paired with the nearest camera frame within a configurable time threshold; unpaired frames are dropped.
- `/tf` transforms are interpolated (linear translation, SLERP rotation) to the exact LiDAR timestamp.
- All point clouds are concatenated into a single flat array. Per-frame `(offset, count)` pairs index into it — no padding, variable point counts supported natively.
- HDF5 datasets are resizable. Camera dimensions and LiDAR pool size are inferred from the first message; `finalize()` trims to the actual sample count.

---

## 2. Installation

Python 3.10+ required.

```bash
pip install mcap2hdf5
```

To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 3. Supported Message Types

| Modality | ROS2 Message Types |
|:---|:---|
| Camera image | `sensor_msgs/msg/CompressedImage`, `sensor_msgs/msg/Image` |
| Camera info | `sensor_msgs/msg/CameraInfo` |
| LiDAR | `sensor_msgs/msg/PointCloud2` |
| Transforms | `tf2_msgs/msg/TFMessage` |

---

## 4. CLI

### 4.1. Inspect an MCAP file

```bash
mcap2hdf5 inspect data.mcap
```

Prints a topic table (topic, message type, message count) and shows which topics would be auto-detected as camera image, camera info, LiDAR, TF, and TF static.

### 4.2. Generate a job config

```bash
mcap2hdf5 init data.mcap
```

Auto-detects sensor topics and writes a YAML job config (`<stem>_config.yaml`) that can be reviewed and edited before running the conversion.

Example output (`data_config.yaml`):

```yaml
source:
  mcap: data.mcap
output:
  hdf5: data.hdf5
modalities:
  camera:
    image_topic: /sensor/camera/left/image_raw/compressed
    info_topic: /sensor/camera/left/camera_info
    sync:
      enabled: true
      algorithm: nearest
      threshold_sec: 0.05
  lidar:
    topic: /sensor/lidar/front/points
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

### 4.3. Convert an MCAP file to HDF5 *(pending)*

```bash
mcap2hdf5 convert data_config.yaml
```

Not yet implemented in the CLI. In the meantime, run the pipeline directly via the example script:

```bash
python3 examples/convert_kitti.py
```

Edit the topic constants at the top of the script to match your sensor configuration. Output is written to `data/processed/chunks.hdf5`.
---

## 5. HDF5 Output Schema

```
chunks.hdf5
│
├── samples/
│   ├── timestamps        [N]           float64  — LiDAR-based reference timestamp (s)
│   └── chunk_ids         [N]           int32    — Temporal chunk index
│
├── camera/
│   └── images            [N, H, W, 3]  uint8    — LZF-compressed RGB frames
│
├── lidar/
│   ├── data              [P, 4]        float32  — Flat point pool: (X, Y, Z, Intensity)
│   ├── offsets           [N]           int64    — Start index of frame i in data
│   └── counts            [N]           int32    — Number of points in frame i
│
├── transforms/
│   └── {frame_id}_to_{child_frame_id}
│                         [N, 4, 4]     float32  — SLERP-interpolated homogeneous matrices
│
├── static_transforms/
│   └── {frame_id}_to_{child_frame_id}
│                         [4, 4]        float32  — Rigid extrinsic transforms
│
└── [File Attributes]
    ├── camera_k          [3, 3]        float32  — Intrinsic matrix
    ├── camera_d          [D]           float32  — Distortion coefficients
    ├── camera_r          [3, 3]        float32  — Rectification matrix
    ├── camera_p          [3, 4]        float32  — Projection matrix
    ├── distortion_model                string
    ├── camera_width / camera_height    int
    ├── num_samples                     int
    └── lidar_point_offset              int
```

Retrieve frame `i` from the LiDAR pool:

```python
offset = hdf5["lidar/offsets"][i]
count  = hdf5["lidar/counts"][i]
points = hdf5["lidar/data"][offset : offset + count]  # shape: (count, 4)
```

---

## 6. Configuration Reference

Parameters are in `mcap2hdf5/configs/`:

| Parameter | Default | Description |
|:---|:---|:---|
| `SENSOR_SYNC_THRESHOLD` | `0.05 s` | Max LiDAR↔Camera time delta for a valid pair |
| `MAX_CHUNK_GAP` | `0.15 s` | Intra-sensor gap that triggers a chunk flush |
| `TF_CACHE_SIZE` | `100` | Rolling window size for TF interpolation |
| `HDF5_WRITE_BATCH_SIZE` | `100` | Samples accumulated before an HDF5 write |
| `INITIAL_LIDAR_CAPACITY` | `1,000,000 pts` | Pre-allocated LiDAR pool (doubles on overflow) |
| `DATA_COMPRESSION_METHOD` | `lzf` | HDF5 compression filter |

---

## 7. Lint and Unit Testing

```bash
# Lint
ruff check .

# Auto-fix
ruff check --fix .

# Tests
pytest
```

---

## 8. Roadmap

| Version | Scope |
|:---|:---|
| **0.1.0** | Single LiDAR + single camera. CLI `inspect`, `init`, `convert` commands. Auto-detect topics. Validate on KITTI. |
| **0.2.0** | Multiple cameras (N cameras, 1 LiDAR). Schema already forward-compatible. |
| **1.0.0+** | Additional modalities: IMU, Radar, GPS, Odometry. |

---

## 9. License

MIT
