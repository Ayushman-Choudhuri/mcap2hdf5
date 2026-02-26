# mcap2hdf5 — Design Document

## Problem

ROS2 MCAP recordings store multi-sensor data (LiDAR, camera, TF) as an interleaved message stream. Using this data for ML training requires re-reading and re-synchronizing the stream on every run, which is slow and non-reproducible. There is no standard, maintained tool that converts MCAP recordings into a format that ML pipelines can consume directly.

## Goal

A CLI tool (`pip install mcap2hdf5`) that converts one or more MCAP recordings into a compressed, randomly-accessible HDF5 dataset. The user runs three commands:

```
mcap2hdf5 inspect recording.mcap        # what topics are in this file?
mcap2hdf5 init recording.mcap -o c.yaml # auto-generate a config
mcap2hdf5 convert recording.mcap \
          -c config.yaml -o dataset.hdf5 # produce the HDF5
```

The output is an HDF5 file where any sample `i` can be retrieved in O(1) by a PyTorch DataLoader with `num_workers > 0`.

## Scope

**V1 includes:** LiDAR (`PointCloud2`), camera (`CompressedImage`, `Image`), TF/TF_static interpolation, multi-sensor support (N lidars, N cameras), YAML config, three CLI commands, PyPI release.

**Explicitly out of scope for V1:** IMU, radar, GPS, non-CDR MCAP encodings, cloud storage output, real-time conversion.

---

## Architecture

```
CLI (cli.py)
    │
    ├── loadConfig()          # YAML → typed dataclasses
    │
    ▼
MCAPSource (reader.py)        # streams StreamMessage per topic
    │
    ▼
SensorDataSynchronizer        # routes to per-sensor buffers,
  (synchronizer.py)           # flushes on chunk gap → SyncGroup
    │
    ▼
HDF5Writer (writer.py)        # calls ModalityConverter per sensor,
                              # writes batches, finalizes schema
```

### Key design decisions

**Reference-sensor sync.** The synchronizer anchors to one sensor (typically the primary LiDAR). Every other sensor is paired to the nearest reference frame within `threshold_ms`. This matches the approach used in NuScenes, Waymo, and KITTI and avoids combinatorial pairwise complexity.

**Flat LiDAR pool per sensor.** Each LiDAR sensor gets a single contiguous `data [P, F]` array. Per-frame `offsets` and `counts` index into it. No padding, O(1) retrieval, variable point counts handled natively.

**Camera intrinsics as group attributes.** Each camera gets a `camera/{name}/` HDF5 group. Intrinsics are stored as attributes on that group, not as flat file attributes. This allows N cameras with independent intrinsics.

**No ROS2 runtime dependency.** `mcap-ros2-support` deserializes CDR messages without a ROS2 install. Critical for adoption in ML environments.

**YAML config, not CLI flags.** Topic names, field lists, and sync parameters are too verbose for flags. A config file is also versionable alongside the recording.

---

## HDF5 Output Schema

```
dataset.hdf5
├── samples/
│   ├── timestamps        [N]         float64
│   └── chunk_ids         [N]         int32
├── lidar/{name}/
│   ├── data              [P, F]      float32
│   ├── offsets           [N]         int64
│   └── counts            [N]         int32
├── camera/{name}/
│   └── images            [N, H, W, 3] uint8  (lzf)
│   (attrs: k, d, r, p, distortion_model, width, height)
├── transforms/
│   └── {frame}_to_{child} [N, 4, 4] float32
├── static_transforms/
│   └── {frame}_to_{child} [4, 4]    float32
└── [file attrs]
    mcap2hdf5_version, created_at, source_file, num_samples
```

---

## Current State

The working proof-of-concept (`generate_dataset.py`) is hardcoded to a single MCAP file, single LiDAR topic, and single camera topic. The core algorithms — streaming, nearest-neighbor sync, SLERP interpolation, flat pool — are correct and proven. V1 is primarily a generalization and packaging effort, not an algorithmic one.

**Completed:**
- PR-01: `pipeline/` moved into installable `mcap2hdf5/` package, `pyproject.toml` added.

---

## Milestones

### Milestone 1 — Package skeleton (weeks 1–2)
Get the repo into a state where a contributor can clone, install, and run CI before any new features land.

- ~~PR-01: Repo restructure into installable package~~ ✓
- PR-02: CI with ruff linting and pytest (matrix: Python 3.10/3.11/3.12)
- PR-03: `PipelineConfig` — YAML parsing with dataclass validation, replaces hardcoded constants

### Milestone 2 — `inspect` and `init` commands (weeks 2–3)
Give users a way to discover topics and generate a starter config without writing any YAML by hand.

- PR-04: Typer CLI skeleton — stub `inspect`, `init`, `convert` commands
- PR-05: `listTopics()` — index-only MCAP scan, no full message streaming
- PR-06: `inspect` command — rich table output of topics, types, message counts, duration
- PR-07: `detectSensors()` — map ROS2 message types to modality candidates
- PR-08: `generateConfigTemplate()` — produce YAML from detected sensors
- PR-09: Wire `init` command

### Milestone 3 — Generalized pipeline (weeks 3–5)
Refactor the hardcoded proof-of-concept into a config-driven, N-sensor pipeline with no behavior change.

- PR-10: Extract `TFCache` and SLERP into `transforms.py`
- PR-11: `LiDARConverter` in `modalities/lidar.py`
- PR-12: `CameraConverter` in `modalities/camera.py`
- PR-13: Parameterize `MCAPSource` — remove hardcoded metadata topics
- PR-14: Generalize `SensorDataSynchronizer` to N sensors from config
- PR-15: Generalize `HDF5Writer` for N LiDAR sensors
- PR-16: Generalize `HDF5Writer` for N camera sensors
- PR-17: Write file-level provenance attributes on finalize

### Milestone 4 — `convert` command (week 5–6)
Wire the generalized pipeline to the CLI and replace `generate_dataset.py`.

- PR-18: Implement `convert` — load config → stream → sync → write → finalize
- PR-19: Per-chunk progress output with `rich.progress`
- PR-20: Completion summary table per sensor

### Milestone 5 — Tests and release (week 6–7)
Close the quality gap before a public announcement.

- PR-21: Synthetic MCAP fixtures (`tests/fixtures/`) for CI
- PR-22: Integration test — full convert pipeline on fixture MCAP
- PR-23: LICENSE, CONTRIBUTING.md, README
- PR-24: PyPI release workflow on `v*` tag push

---

## Open Questions

- **`dataset.py` / `KittiHDF5Dataset`** — out of scope for the tool itself, but worth offering as `mcap2hdf5.datasets` or a companion package once the schema is stable. Decision deferred to post-V1.
- **`requirements.txt`** — kept for now to avoid breaking existing users. Remove in PR-02 once `pyproject.toml` is the single source of truth for dependencies.
