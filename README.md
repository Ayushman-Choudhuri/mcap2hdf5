# mcap2ml : A cli tool to convert raw robot data into ML datasets. 

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/ROS2-MCAP-22314E?style=flat-square&logo=ros&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Storage-HDF5%20%2B%20LZF-FF0000?style=flat-square"/>
  <img src="https://img.shields.io/badge/Data-LiDAR%20%2B%20Camera%20%2B%20TF-brightgreen?style=flat-square"/>
</p>

<p align="center">
  A streaming ETL pipeline that converts raw ROS2 multi-modal robotics telemetry (MCAP) into temporally-synchronized, O(1)-access HDF5 datasets — ready to plug directly into a PyTorch training loop.
</p>

---

## The Problem

Training autonomous driving and robotics perception models requires pairing raw sensor data — LiDAR point clouds, camera images, and coordinate frame transforms — into synchronized, indexed samples. ROS2 bags (MCAP format) store this data as an interleaved stream of typed messages. Naively iterating this stream during training is slow, memory-intensive, and blocks GPU utilization.

This pipeline solves that by doing all the heavy lifting once: reading, temporally synchronizing, interpolating transforms, and writing a compressed, randomly-accessible HDF5 dataset that DataLoaders can index into in O(1).

---

## Key Features

- **Zero-copy streaming** — `MCAPSource` is a generator that reads one message at a time, keeping memory usage flat regardless of file size.
- **Temporal synchronization** — each LiDAR frame is paired with the nearest camera frame within a configurable time threshold (`SENSOR_SYNC_THRESHOLD = 50 ms`). Unpaired frames are discarded.
- **Chunk-aware buffering** — messages are grouped into temporal chunks (gap detection via `MAX_CHUNK_GAP = 150 ms`), preventing cross-sequence pairings at recording boundaries.
- **SLERP transform interpolation** — `/tf` transforms are cached in a rolling window and interpolated to the exact LiDAR timestamp using Spherical Linear Interpolation, avoiding gimbal lock.
- **Flat LiDAR pool** — point clouds are concatenated into a single `(P, 4)` array. Per-frame `offsets` and `counts` index into it, eliminating padding and supporting variable point counts natively.
- **O(1) random access** — the HDF5 schema is designed so that any sample `i` can be retrieved in constant time by any number of DataLoader workers simultaneously (SWMR mode).
- **Persisted sensor metadata** — camera intrinsics (K, D, R, P matrices), distortion model, and static extrinsic transforms are written as HDF5 file attributes at finalization.
- **PyTorch-native** — `KittiHDF5Dataset` implements `__len__`/`__getitem__` with lazy file handles (safe for `num_workers > 0`). A custom `kitti_collate_fn` handles variable-length LiDAR tensors in batches.

---

## HDF5 Output Schema

```
chunks.hdf5
│
├── samples/
│   ├── timestamps            [N]        float64  — LiDAR-based reference timestamp (seconds)
│   └── chunk_ids             [N]        int32    — Temporal chunk index
│
├── camera/
│   └── images                [N, H, W, 3]  uint8   — LZF-compressed RGB frames
│
├── lidar/
│   ├── data                  [P, 4]     float32  — Flat point pool: (X, Y, Z, Intensity)
│   ├── offsets               [N]        int64    — Start index of frame i in data
│   └── counts                [N]        int32    — Number of points in frame i
│
├── transforms/
│   └── {frame_id}_to_{child_frame_id}
│                             [N, 4, 4]  float32  — SLERP-interpolated homogeneous matrices
│
└── static_transforms/
    └── {frame_id}_to_{child_frame_id}
                              [4, 4]     float32  — Rigid extrinsic transforms
│
└── [File Attributes]
    ├── camera_k              [3, 3]     float32  — Camera intrinsic matrix
    ├── camera_d              [D]        float32  — Distortion coefficients
    ├── camera_r              [3, 3]     float32  — Rectification matrix
    ├── camera_p              [3, 4]     float32  — Projection matrix
    ├── distortion_model               string
    ├── camera_width / camera_height   int
    ├── num_samples                    int
    └── lidar_point_offset             int
```

Retrieving sample `i` from the LiDAR pool:
```python
offset = hdf5["lidar/offsets"][i]
count  = hdf5["lidar/counts"][i]
points = hdf5["lidar/data"][offset : offset + count]  # shape: (count, 4)
```

---

## Quick Start

**1. Install dependencies** (Python 3.10+ recommended)
```bash
pip install -r requirements.txt
```

**2. Place your MCAP file**
```
data/raw/kitti.mcap
```
Topic names and file paths are configured in `pipeline/config.py`.

**3. Run the pipeline**
```bash
python3 generate_dataset.py
```
Output is written to `data/processed/chunks.hdf5`. The pipeline logs progress per batch and prints a summary on completion:
```
Dataset Statistics:
  Total samples: 1847
  Total lidar points: 52,184,291
  Average points per sample: 28,255.2
```

**4. Verify with the PyTorch DataLoader**
```bash
python3 dataset.py
```
```
Batch loaded: Image shape torch.Size([8, 3, H, W]), LiDAR frames in batch: 8
```

---

## Configuration

All pipeline parameters are centralized in `pipeline/config.py`:

| Parameter | Default | Description |
|:---|:---|:---|
| `MCAP_FILE_PATH` | `data/raw/kitti.mcap` | Input MCAP file |
| `CHUNKS_FILE_PATH` | `data/processed/chunks.hdf5` | Output HDF5 file |
| `SENSOR_SYNC_THRESHOLD` | `0.05 s` | Max LiDAR↔Camera time delta for a valid pair |
| `MAX_CHUNK_GAP` | `0.15 s` | Intra-sensor gap that triggers a chunk flush |
| `TF_CACHE_SIZE` | `100` | Rolling window size for TF interpolation |
| `HDF5_WRITE_BATCH_SIZE` | `100` | Samples accumulated before an HDF5 write |
| `INITIAL_LIDAR_CAPACITY` | `1,000,000 pts` | Pre-allocated LiDAR pool (doubles on overflow) |
| `DATA_COMPRESSION_METHOD` | `lzf` | HDF5 compression filter |

ROS2 topic names (`LIDAR_TOPIC`, `CAMERA_IMAGE_TOPIC`, etc.) are also defined there — update them to match your sensor configuration.

---

## How It Works

### Temporal Synchronization
Messages arrive interleaved from multiple topics. The synchronizer maintains a separate buffer per sensor and uses **LiDAR timestamp as the reference**: for each LiDAR frame, the camera frame with the smallest absolute time difference is selected. If the difference exceeds `SENSOR_SYNC_THRESHOLD`, the LiDAR frame is dropped.

### Chunk Segmentation
A flush is triggered when consecutive messages on the same topic are more than `MAX_CHUNK_GAP` apart. This segments the recording into temporal chunks (e.g., separate drives or episodes), preventing the synchronizer from pairing frames across a temporal gap.

### Transform Interpolation
`/tf` messages are cached in a per-key rolling list. For each synchronized sample, transforms are interpolated to the LiDAR timestamp: **linear** for translation, **SLERP** (quaternion) for rotation — avoiding the artifacts of naive matrix interpolation.

### LiDAR Storage
Variable point counts per frame are handled without padding: all points are concatenated into a single flat array, and per-frame `(offset, count)` pairs act as an index. This keeps storage dense and retrieval O(1).

---
