# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# Run the ETL pipeline (MCAP → HDF5)
python3 generate_dataset.py

# Verify the HDF5 output with a PyTorch DataLoader test
python3 dataset.py
```

There is no test suite or linter configured. The devcontainer (`/.devcontainer/`) uses Docker Compose and mounts the repo at `/workspace`.

## Architecture

This is a streaming ETL pipeline that converts ROS2 MCAP recordings into HDF5 datasets for ML training. Data flows through three stages orchestrated by `generate_dataset.py`:

```
MCAPSource → SensorDataSynchronizer → HDF5Writer
```

### pipeline/reader.py — `MCAPSource`
A generator that streams one message at a time from an MCAP file using `mcap_ros2`. It captures camera intrinsics (first `CameraInfo` message) and static TF transforms as side-effects during the stream, retrievable via `getCameraMetadata()` / `getStaticTransforms()` after streaming completes. Timestamps are extracted preferentially from `header.stamp`, falling back to MCAP log time.

### pipeline/synchronizer.py — `SensorDataSynchronizer`
Maintains per-topic buffers (`LIDAR_TOPIC`, `CAMERA_IMAGE_TOPIC`) and a rolling TF cache. Key behaviors:
- **Chunk flushing**: when a gap > `MAX_CHUNK_GAP` is detected on any sensor topic, `flushSamples()` is called, pairing all buffered LiDAR frames with their nearest camera frame. Pairs exceeding `SENSOR_SYNC_THRESHOLD` are discarded.
- **TF interpolation**: `/tf` messages are cached per `{frame_id}_to_{child_frame_id}` key (capped at `TF_CACHE_SIZE`). For each synced sample, transforms are interpolated to the LiDAR timestamp using linear translation + SLERP rotation (`MessageConverter.interpolateMatrix`).
- The generator in `generate_dataset.py` uses `flushEventTriggered` to increment `chunkId` — note this triggers on any flush, including inter-chunk boundaries.

### pipeline/hdf5_writer.py — `HDF5Writer`
Writes batched samples to HDF5 with resizable datasets. Key design:
- **Flat LiDAR pool**: all point clouds are concatenated into a single `lidar/data [P, 4]` array. Per-frame `offsets` and `counts` index into it. The pool starts at `INITIAL_LIDAR_CAPACITY` and doubles when full.
- Datasets are created on first batch (lazily) so camera image dimensions are inferred from the first real sample.
- `finalize()` trims the LiDAR pool to actual size and writes camera intrinsics + static transforms as HDF5 file attributes/groups. Must always be called — if not, `__del__` logs a warning and closes the file without trimming.

### pipeline/message_converter.py — `MessageConverter`
Static methods for ROS2 message → numpy conversion:
- `lidarToNumpy`: parses `PointCloud2` binary data using field dtype mapping, extracts `(x, y, z, intensity)` → `(N, 4) float32`
- `compressedImageToNumpy`: decodes JPEG/PNG bytes via OpenCV → `(H, W, 3) uint8`
- `transformToMatrix` / `interpolateMatrix`: quaternion → 4×4 homogeneous matrix; SLERP interpolation with dot-product sign correction

### dataset.py — `KittiHDF5Dataset`
PyTorch `Dataset` with lazy file handle initialization (safe for `num_workers > 0`, using SWMR mode). `kitti_collate_fn` handles variable-length LiDAR tensors by keeping them as a list rather than stacking.

### pipeline/config.py
Single source of truth for all file paths, topic names, thresholds, and HDF5 dataset path strings. **Update topic names here** when adapting to a different sensor configuration.

## Data Layout

Input: `data/raw/kitti.mcap`
Output: `data/processed/chunks.hdf5`

Retrieving LiDAR for sample `i`:
```python

offset = hdf5["lidar/offsets"][i]
count  = hdf5["lidar/counts"][i]
points = hdf5["lidar/data"][offset : offset + count]  # (count, 4) float32
```
### Code Style
- Variables: camelCase
- Functions and methods: camelCase  
- Class names: PascalCase
- File names: snake_case
- Constants: SCREAMING_SNAKE_CASE
- Imports: alphabetical order within groups (stdlib, third-party, local)
- Comments: Minimal and only when necessary. Use docstrings (""" """) for modules, classes, and functions. Use # for inline comments.
