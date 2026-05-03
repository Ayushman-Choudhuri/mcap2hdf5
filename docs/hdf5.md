# HDF5 File Format

## What is HDF5

HDF5 (Hierarchical Data Format version 5) is a binary file format designed for storing and organizing large, complex scientific datasets. It was developed by the HDF Group and is widely used in scientific computing, astronomy, climate science, and increasingly in ML — particularly for multimodal training data.

The format provides two core abstractions:
- **Groups** — containers, analogous to directories in a filesystem
- **Datasets** — typed, N-dimensional arrays, analogous to files

---

## File Structure

An HDF5 file is a self-describing hierarchical container. The structure is best understood as a filesystem tree embedded in a single binary file:

```
output.hdf5
│
├── samples/                         GROUP
│   ├── timestamps   [N]  float64    DATASET — Unix epoch seconds per frame
│   └── chunk_ids    [N]  int32      DATASET — which recording chunk each frame came from
│
├── camera/                          GROUP
│   └── images  [N, H, W, 3]  uint8 DATASET — compressed with lzf
│
├── lidar/                           GROUP
│   ├── data     [P, 4]  float32    DATASET — flat pool: all points x,y,z,intensity
│   ├── offsets  [N]     int64      DATASET — start index in data[] for frame i
│   └── counts   [N]     int32      DATASET — number of points for frame i
│
├── transforms/                      GROUP
│   └── odom_to_lidar_frame  [N, 4, 4]  float32   DATASET — per-frame transform
│
├── static_transforms/               GROUP
│   └── lidar_frame_to_camera_frame  [4, 4]  float32   DATASET — fixed calibration
│
└── [File Attributes]                KEY-VALUE METADATA
    ├── num_samples          1067
    ├── lidar_point_offset   112700642
    ├── camera_k             [3, 3]    intrinsic matrix
    ├── camera_d             [5]       distortion coefficients
    ├── camera_r             [3, 3]    rectification matrix
    ├── camera_p             [3, 4]    projection matrix
    ├── camera_width         1241
    ├── camera_height        376
    └── distortion_model     plumb_bob
```

### Key structural concepts

**Groups** are pure containers. They have no data themselves but can hold datasets and other groups. They support arbitrary nesting. In mcap2hdf5 groups are used to separate modalities (`camera/`, `lidar/`, `transforms/`) so the file structure mirrors the sensor rig.

**Datasets** are typed, N-dimensional arrays with a declared shape, dtype, and optional compression. They support:
- Fixed or resizable shapes (`maxshape=(None, ...)`)
- Chunked storage — data is split into fixed-size tiles on disk for efficient partial reads
- Per-chunk compression (lzf, gzip, zstd)
- Partial I/O — read a slice without loading the full dataset

**Attributes** are small key-value pairs attached to any group or dataset (or the root file). Used for metadata that describes the data rather than being the data itself — calibration matrices, version strings, topic names.

---

## Why HDF5 for Multimodal ML Training

### The core problem it solves

Multimodal training data (camera + LiDAR + IMU + transforms) consists of arrays with very different shapes, dtypes, and access patterns. Storing them as separate files (one `.npy` per frame, one `.jpg` per image) creates millions of small files that overwhelm filesystems and make random access slow. HDF5 solves this by packing everything into one file while preserving random access to individual frames.

### Specific advantages for robotics ML

**1. Random access by frame index**
A DataLoader can read frame `i` directly: `images[i]`, `lidar_data[offsets[i]:offsets[i]+counts[i]]`. No sequential scan needed. This is what makes shuffling during training efficient.

**2. Memory-mapped I/O**
h5py can open a file in SWMR (Single Writer Multiple Reader) mode and memory-map datasets. Workers in a multi-process DataLoader each get their own file handle with no locking overhead. This is why `KittiHDF5Dataset._init_db` is lazy — each worker initializes its own handle after forking.

**3. Heterogeneous arrays in one file**
Camera images (`uint8`), LiDAR points (`float32`), transforms (`float32`), timestamps (`float64`) all live in one file with consistent indexing. A single `dataset[i]` call retrieves the correct frame across all modalities.

**4. Compression without full decompression**
Chunked datasets decompress only the chunks that contain the requested slice. Reading one image frame decompresses one chunk, not the entire images dataset.

**5. Self-describing**
Datasets carry their own shape and dtype. Attributes store calibration. A reader doesn't need an external schema file to understand the data.

**6. Partial writes and resizing**
`maxshape=(None, ...)` allows datasets to grow during the conversion run. `HDF5Writer` uses this to accumulate samples in batches without knowing the total count upfront.

---

## Advantages vs Other Formats

| Property | HDF5 | NumPy `.npy` files | JPEG/PNG + CSV | WebDataset (tar) |
|----------|------|-------------------|----------------|-----------------|
| Random access | O(1) by index | O(1) per file, but millions of files | O(1) per file | Sequential only |
| Heterogeneous arrays | Yes — one file | No — one file per array | No | Yes — mixed entries |
| Compression | Per-chunk, transparent | Optional, whole-array | JPEG lossy / PNG lossless | Per-file |
| Multi-worker DataLoader | Yes — SWMR mode | Yes | Yes | Yes — sharding |
| Streaming (large datasets) | Partial — chunked reads | No | No | Yes — designed for it |
| Human inspectable | No — binary | No | Partially | No |
| Ecosystem support | h5py, PyTorch, TF, MATLAB, R | NumPy, PyTorch | Universal | PyTorch, HuggingFace |
| Schema evolution | Manual | None | None | None |

---

## Disadvantages

**Not designed for distributed training at scale**
HDF5 is excellent for a single machine. For multi-node training (thousands of GPUs), WebDataset or tfrecord shards are better because they are designed for streaming from object storage (S3, GCS). HDF5 random access requires seekable file I/O which doesn't map cleanly to object storage.

**Write contention**
Only one writer at a time. Parallel conversion jobs must write to separate files. There is no native merge operation.

**No schema enforcement**
Adding a new dataset to the file doesn't break old readers, but removing one does. There is no migration tooling — version management is manual (hence the `mcap2hdf5_version` attribute pattern).

**Fragmentation over time**
Repeated resize-and-shrink operations leave dead space in the file. `h5repack` can defragment but requires a full file copy.

**Not human-readable**
You cannot `cat` or `grep` an HDF5 file. Inspection requires tooling (`h5py`, `h5ls`, `HDFView`, or the `inspect_hdf5.py` script).

---

## The Flat LiDAR Pool Pattern

Variable-length point clouds are the hardest multimodal storage problem. Each LiDAR frame has a different number of points, so you cannot store them as a uniform `[N, max_points, 4]` array without padding — padding wastes enormous space and corrupts statistics.

The flat pool pattern solves this:

```
lidar/data     [P, 4]   — all points from all frames concatenated
lidar/offsets  [N]      — offsets[i] = start index of frame i in data
lidar/counts   [N]      — counts[i]  = number of points in frame i
```

Retrieving frame `i`:
```python
start = offsets[i]
end   = start + counts[i]
points = data[start:end]   # shape: [counts[i], 4]
```

This is O(1) random access with zero padding. The pool starts at `INITIAL_LIDAR_CAPACITY` and doubles when full (`resizeLidarData`) — the same growth strategy as a dynamic array.

The tradeoff: the pool must be trimmed at the end of the conversion run. `finalize()` calls `resize` to cut `data` down to the actual number of points written. If `finalize()` is not called, the file contains uninitialized zeros at the end.

---

## SWMR Mode and Multi-Worker DataLoaders

PyTorch DataLoader with `num_workers > 0` forks multiple processes. Each worker must open its own file handle — sharing one handle across processes causes corruption. The standard pattern:

```python
class KittiHDF5Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.db = None          # not opened at construction

    def _init_db(self):
        if self.db is None:
            self.db = h5py.File(self.path, "r", swmr=True)

    def __getitem__(self, idx):
        self._init_db()         # each worker opens on first access after fork
        ...
```

SWMR (Single Writer Multiple Reader) mode allows multiple readers to open the same file concurrently without locking. Opening at construction time would cause the handle to be shared across forked workers — lazy initialization ensures each worker gets its own.

---

## Interview Questions

**Why HDF5 over a directory of NumPy files for a 1M frame dataset?**
A directory of 1M `.npy` files (one per frame, one per modality) creates 3–5M filesystem inodes. Most Linux filesystems degrade significantly above ~1M files in a directory. `ls`, `find`, and DataLoader workers that open files per sample all become slow. HDF5 packs everything into one file with O(1) random access by index — the DataLoader reads a slice, not a file.

**How does HDF5 handle compression without loading the full dataset?**
Data is stored in fixed-size chunks (e.g. 128×128 tiles for images). Each chunk is independently compressed. Reading a single element decompresses only the chunk containing it, not the whole dataset. Chunk size is a tuning parameter — small chunks reduce read amplification for random access, large chunks improve sequential throughput and compression ratio.

**What is the difference between HDF5 attributes and datasets?**
Attributes are small key-value pairs for metadata — calibration matrices, version strings, topic names. Datasets are the primary data arrays. The practical rule: if it describes the data, use an attribute. If it is the data, use a dataset. Attributes are not chunkable or compressible and should not hold large arrays.

**What breaks if `finalize()` is not called on HDF5Writer?**
Three things: (1) the LiDAR data pool is not trimmed — the file contains trailing zeros beyond the actual data. (2) datasets are not resized to the true sample count. (3) camera intrinsics and static transforms are not written to file attributes. The file is structurally valid but contains garbage in the LiDAR tail and is missing all calibration metadata.

**Why is the LiDAR dataset `[P, 4]` instead of `[N, max_points, 4]`?**
Padding to `max_points` wastes space proportional to `(max_points - mean_points) × N × 4 bytes`. For a Velodyne HDL-64E, max points per spin is ~130k but mean is ~105k — that's ~25% wasted space at 4 bytes per float. More critically, padded zeros corrupt per-frame statistics (mean intensity, point count distributions) used in data augmentation and normalization. The flat pool with offsets/counts has zero waste and zero corruption.

**How do you handle schema changes between dataset versions?**
Store a version attribute (`mcap2hdf5_version`) in the file. Readers check the version at load time and apply any necessary transformations. For breaking changes (renamed paths, changed dtypes), bump the major version and provide a migration script that reads the old schema and writes a new file. HDF5 has no built-in migration — version management is entirely the application's responsibility.

**What is the difference between resizable and fixed datasets?**
A fixed dataset has a declared shape that cannot change after creation. A resizable dataset is created with `maxshape=(None, ...)` and can grow via `.resize()`. Resizable datasets require chunked storage — HDF5 cannot resize a contiguous allocation. In mcap2hdf5, all datasets are resizable because the total sample count is not known until the conversion run completes.
