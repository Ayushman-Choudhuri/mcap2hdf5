# mcap2hdf5 — V1 Prototype Plan

## Project Goal

An open-source command-line utility that converts multimodal ROS2 MCAP recordings into compressed, randomly-accessible HDF5 datasets ready for ML pipelines. V1 targets camera and LiDAR modalities. IMU and radar come later.

The existing codebase is a working proof-of-concept but is hardcoded to a single MCAP file with fixed topic names. The work for V1 is primarily about generalizing it into a properly packaged, user-configurable CLI tool.

---

## What's In / Out for V1

**In scope:**
- LiDAR (`sensor_msgs/PointCloud2`) — one or more sensors
- Camera — compressed (`sensor_msgs/CompressedImage`) and raw (`sensor_msgs/Image`) — one or more sensors
- TF and TF_static interpolation (existing behavior, generalized)
- YAML-based configuration (topic names, sync parameters, output settings)
- Three CLI commands: `inspect`, `init`, `convert`
- Installable Python package published to PyPI (`pip install mcap2hdf5`)
- Multiple concurrent LiDAR and camera sensors in a single recording

**Out of scope for V1:**
- IMU, radar, GPS modalities
- Non-ROS2 MCAP encodings (CDR only for now)
- Real-time / streaming conversion
- Cloud storage output (S3, GCS)
- GUI or web interface

---

## User Experience

### Installation
```bash
pip install mcap2hdf5
```

### Typical workflow

**Step 1 — Discover what's in the recording:**
```bash
mcap2hdf5 inspect recording.mcap
```
```
Topics in recording.mcap (duration: 4m 32s, 12,847 messages):

  Topic                                      Type                              Messages
  ─────────────────────────────────────────────────────────────────────────────────────
  /sensor/lidar/front/points                 sensor_msgs/PointCloud2           2,741
  /sensor/lidar/rear/points                  sensor_msgs/PointCloud2           2,739
  /sensor/camera/left/image_raw/compressed   sensor_msgs/CompressedImage       5,482
  /sensor/camera/right/image_raw/compressed  sensor_msgs/CompressedImage       5,476
  /sensor/camera/left/camera_info            sensor_msgs/CameraInfo            5,482
  /tf                                        tf2_msgs/TFMessage                8,201
  /tf_static                                 tf2_msgs/TFMessage                1
  /imu/data                                  sensor_msgs/Imu                   27,140
```

**Step 2 — Generate a starter config:**
```bash
mcap2hdf5 init recording.mcap -o config.yaml
```
This auto-detects topics by message type and writes a YAML template with every detected sensor pre-populated. The user edits it to keep only what they want.

**Step 3 — Convert:**
```bash
mcap2hdf5 convert recording.mcap -c config.yaml -o dataset.hdf5
```
```
[mcap2hdf5] Reading recording.mcap ...
[mcap2hdf5] Chunk  1/? — written 100 samples (lidar/front: 100, camera/left: 100, camera/right: 100)
[mcap2hdf5] Chunk  2/? — written 100 samples ...
...
[mcap2hdf5] Finalized dataset.hdf5

  Modality        Sensor    Samples    Notes
  ──────────────────────────────────────────────────────
  lidar           front     1,847      52.2M points total
  lidar           rear      1,845      49.1M points total
  camera          left      1,847      1920×1080, lzf
  camera          right     1,847      1920×1080, lzf
  transforms                1,847      3 dynamic TF keys
  static_transforms         —          2 static TF keys
```

---

## Configuration Schema

Config is YAML. The `init` command generates this; users edit it.

```yaml
version: 1

sync:
  reference: lidar/front        # sensor whose timestamps drive pairing
  threshold_ms: 50              # max lidar↔camera time diff for a valid pair
  chunk_gap_ms: 150             # intra-sensor gap that triggers a chunk flush
  tf_cache_size: 100            # rolling window for TF interpolation

sensors:
  lidar:
    - name: front               # used as HDF5 group name and in logs
      topic: /sensor/lidar/front/points
      fields: [x, y, z, intensity]   # PointCloud2 fields to extract

    - name: rear
      topic: /sensor/lidar/rear/points
      fields: [x, y, z, intensity]

  camera:
    - name: left
      topic: /sensor/camera/left/image_raw/compressed
      info_topic: /sensor/camera/left/camera_info   # optional; for intrinsics

    - name: right
      topic: /sensor/camera/right/image_raw/compressed
      info_topic: /sensor/camera/right/camera_info

output:
  compression: lzf              # lzf | gzip | none
  write_batch_size: 100
  initial_lidar_capacity: 1000000   # pre-allocated points per lidar sensor
```

The `sync.reference` field identifies which sensor is the anchor for temporal pairing. Every other sensor is paired to the nearest sample of the reference sensor within `threshold_ms`.

---

## Software Architecture

### Package structure

```
mcap2hdf5/
├── pyproject.toml
├── mcap2hdf5/
│   ├── __init__.py
│   ├── cli.py                  # Typer app, entry point
│   ├── config.py               # YAML parsing → typed dataclasses
│   ├── reader.py               # MCAPSource (generalized)
│   ├── synchronizer.py         # SensorDataSynchronizer (generalized)
│   ├── writer.py               # HDF5Writer (generalized)
│   ├── modalities/
│   │   ├── __init__.py
│   │   ├── base.py             # ModalityConverter abstract base
│   │   ├── lidar.py            # PointCloud2 → numpy
│   │   └── camera.py           # CompressedImage / Image → numpy
│   └── transforms.py           # TF cache, SLERP interpolation
└── tests/
    ├── fixtures/               # small synthetic MCAP files for tests
    ├── test_config.py
    ├── test_synchronizer.py
    ├── test_modalities.py
    └── test_writer.py
```

### Data flow

```
CLI (cli.py)
    │
    ├── loads Config (config.py)
    │
    ▼
MCAPSource (reader.py)
    │  streams StreamMessage(topic, msg, timestamp) for ALL topics
    ▼
SensorDataSynchronizer (synchronizer.py)
    │  routes messages to per-sensor buffers
    │  flushes on chunk gap → yields SyncGroup
    ▼
HDF5Writer (writer.py)
    │  calls ModalityConverter per sensor per sample
    │  writes batches to HDF5
    ▼
output.hdf5
```

### Layer responsibilities

**`cli.py`** — three Typer commands: `inspect`, `init`, `convert`. No business logic here; delegates entirely to the other layers. `inspect` and `init` only use `MCAPSource` directly.

**`config.py`** — parses and validates the YAML config into typed dataclasses (`PipelineConfig`, `SyncConfig`, `LiDARSensorConfig`, `CameraSensorConfig`). Raises clear errors on missing required fields. Provides defaults for all optional fields.

**`reader.py` — `MCAPSource`** — unchanged in behavior from the current code but generalized: it no longer hardcodes which topics to watch for metadata. Instead it accepts a set of `info_topics` from the config and captures the first message on each as metadata. It yields `StreamMessage` for every topic without filtering.

**`modalities/base.py` — `ModalityConverter`** — abstract base with one required method:
```python
class ModalityConverter(ABC):
    @abstractmethod
    def toNumpy(self, rosMsg) -> np.ndarray:
        ...
```

**`modalities/lidar.py` — `LiDARConverter`** — takes the configured `fields` list and extracts those columns from `PointCloud2`. Raises `ValueError` if the message is missing a required field.

**`modalities/camera.py` — `CameraConverter`** — handles both `sensor_msgs/CompressedImage` (JPEG/PNG decode via OpenCV) and `sensor_msgs/Image` (raw buffer reshape). Encoding is detected from the message, not the config.

**`transforms.py`** — the TF cache and SLERP interpolation logic extracted from the current `synchronizer.py` and `message_converter.py`. Isolated here because it has no sensor-specific logic and benefits from unit testing independently.

**`synchronizer.py` — `SensorDataSynchronizer`** — generalized to handle N sensors defined by the config. The reference sensor is specified by `sync.reference`. For each chunk flush:
1. Iterate reference sensor frames
2. For every other sensor, find its nearest frame to the reference timestamp
3. Discard the pair if any sensor exceeds `threshold_ms`
4. Interpolate TF transforms to the reference timestamp

This is an extension of the existing nearest-neighbor matching approach. A `SyncGroup` maps sensor name → entry dict.

**`writer.py` — `HDF5Writer`** — generalized to write N lidar sensors and N camera sensors. Each sensor gets its own HDF5 group (`lidar/{name}/`, `camera/{name}/`). Each LiDAR sensor has its own flat point pool with its own offset/count index. Camera intrinsics are stored under `camera/{name}/` group attributes rather than flat file attributes (so multiple cameras can each have their own intrinsics).

---

## HDF5 Output Schema (V1)

```
dataset.hdf5
│
├── samples/
│   ├── timestamps              [N]           float64   LiDAR reference timestamps (s)
│   └── chunk_ids               [N]           int32     Temporal chunk index
│
├── lidar/
│   ├── front/
│   │   ├── data                [P_f, F]      float32   Flat point pool; F = len(fields)
│   │   ├── offsets             [N]           int64     Start index in data for sample i
│   │   └── counts              [N]           int32     Points in sample i
│   └── rear/
│       ├── data                [P_r, F]      float32
│       ├── offsets             [N]           int64
│       └── counts              [N]           int32
│
├── camera/
│   ├── left/
│   │   └── images              [N, H, W, 3]  uint8     LZF-compressed RGB
│   │   (group attrs: k, d, r, p, distortion_model, width, height)
│   └── right/
│       └── images              [N, H, W, 3]  uint8
│       (group attrs: k, d, r, p, distortion_model, width, height)
│
├── transforms/
│   └── {frame_id}_to_{child}   [N, 4, 4]    float32   SLERP-interpolated
│
├── static_transforms/
│   └── {frame_id}_to_{child}   [4, 4]       float32   Rigid extrinsics
│
└── [File Attributes]
    ├── mcap2hdf5_version        string        e.g., "0.1.0"
    ├── created_at               string        ISO 8601
    ├── source_file              string        original MCAP filename
    └── num_samples              int
```

The `samples/timestamps` array always reflects the reference sensor's timestamps. All other sensor arrays are indexed by the same `N` — they are all guaranteed to have `N` entries corresponding to the same `N` synchronized moments in time.

---

## Key Technical Decisions

**Typer for CLI, not argparse.** Typer generates `--help` text automatically from type annotations, handles type coercion, and produces shell completion scripts out of the box. It is the right level of abstraction for a utility that will be maintained by contributors.

**YAML config, not CLI flags for sensor topology.** Topic names, field lists, and sensor names are too verbose for flags. A config file is also versionable alongside the recording, which is important when re-processing datasets months later.

**Reference-sensor synchronization.** The synchronizer stays anchored to the reference sensor (typically the primary LiDAR). Every other sensor is paired to the reference. This avoids the combinatorial complexity of pairwise multi-sensor sync and matches the established practice in AV datasets (NuScenes, Waymo, KITTI all use one primary sensor as the temporal anchor).

**Per-sensor flat LiDAR pools.** Each LiDAR sensor keeps its own pool rather than a combined one. This keeps retrieval straightforward and avoids ambiguity about which points belong to which sensor.

**`mcap2hdf5 init` auto-detection.** The `init` command maps ROS2 message types to modalities:
- `sensor_msgs/PointCloud2` → lidar candidate
- `sensor_msgs/CompressedImage` or `sensor_msgs/Image` → camera candidate
- `sensor_msgs/CameraInfo` → paired to the camera with the matching topic prefix

Topics for IMU (`sensor_msgs/Imu`) and radar (`sensor_msgs/PointCloud2` from a radar namespace) are listed in `inspect` output but omitted from the generated config with a comment: `# IMU/radar support coming in v2`.

**Packaging as a proper Python package with `pyproject.toml`.** The entry point `mcap2hdf5 = "mcap2hdf5.cli:app"` makes the CLI available after `pip install`. Publishing to PyPI is the primary distribution channel for the research audience.

**No runtime dependency on ROS2.** `mcap-ros2-support` deserializes CDR-encoded messages without needing a ROS2 installation. This is critical for adoption — ML engineers should be able to run the tool on a laptop or in a Docker container without installing ROS2.

---

## Refactoring the Existing Code

The gap between the current codebase and the target architecture is smaller than it looks. The core algorithms (streaming, sync, SLERP, flat LiDAR pool) are correct and proven. The work is generalization:

| Current | Change needed |
|:--------|:--------------|
| `pipeline/config.py` — hardcoded constants | Replace with YAML-parsed dataclasses in `config.py` |
| `pipeline/reader.py` — captures metadata for hardcoded topics | Parameterize: accept a set of info_topics to watch |
| `pipeline/synchronizer.py` — hardcoded `LIDAR_TOPIC`, `CAMERA_IMAGE_TOPIC` | Generalize buffers to a dict keyed by sensor name from config |
| `pipeline/message_converter.py` — mixed concerns | Split: LiDAR → `modalities/lidar.py`, Camera → `modalities/camera.py`, TF → `transforms.py` |
| `pipeline/hdf5_writer.py` — single camera, single lidar, flat file attributes for intrinsics | Generalize groups; camera intrinsics become group attributes |
| `generate_dataset.py` — orchestration script | Becomes the body of the `convert` CLI command |
| No CLI | `cli.py` with Typer |

The existing `dataset.py` (`KittiHDF5Dataset`) is out of scope for V1 of the tool itself, but should be offered as a separate `mcap2hdf5.datasets` module (or a companion package) once the schema is stable.

---

## Open Source Setup

**License:** MIT. Maximally permissive for research lab adoption.

**Minimum repo structure at launch:**
```
README.md            — quickstart, install, config reference, schema reference
CONTRIBUTING.md      — dev setup, how to add a new modality, PR process
CHANGELOG.md         — versioned changes
LICENSE
pyproject.toml       — build system, dependencies, entry point, version
.github/
  workflows/
    ci.yaml          — pytest + ruff on push/PR, across Python 3.10/3.11/3.12
```

**Testing strategy:** The core risk is message decoding (LiDAR field parsing, image decoding) and synchronization logic. Unit tests should use small synthetic MCAP files checked into `tests/fixtures/`, generated once with a helper script and committed as binary fixtures. This avoids requiring large real recordings in CI.

**First target communities for feedback:**
- ROS2 research groups converting their own bags for training
- Autonomous driving researchers using KITTI/NuScenes-style data in custom recordings
- Robotics ML teams at universities

---

## Development Milestones

**Milestone 1 — Installable package skeleton**
- `pyproject.toml` with entry point, dependencies
- `cli.py` with stub `inspect`, `init`, `convert` commands
- `config.py` with YAML parsing and validation
- CI workflow (ruff lint + pytest)

**Milestone 2 — `inspect` and `init` commands**
- `MCAPSource.listTopics()` — scan without full streaming, return topic stats
- Message-type → modality auto-detection for `init`
- YAML template generation

**Milestone 3 — Generalized pipeline**
- Split `message_converter.py` → `modalities/lidar.py`, `modalities/camera.py`, `transforms.py`
- Generalize `synchronizer.py` to N sensors from config
- Generalize `writer.py` to N sensors with the new schema

**Milestone 4 — `convert` command end-to-end**
- Wire config → reader → synchronizer → writer in `cli.py`
- Progress reporting (samples written per chunk, per sensor)
- Summary table on completion

**Milestone 5 — Tests and documentation**
- Unit tests for config parsing, modality converters, synchronizer, writer
- README with full config reference and schema reference
- v0.1.0 PyPI release
