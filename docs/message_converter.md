# message_converter.py — Design Notes

---

## 1.0 Overview

`MessageConverter` provides static conversion utilities from ROS2 message types to numpy arrays. Each function in the class has a dedicated section below covering its background, implementation, and API decisions.

| Function                 | Input                              | Output                  |
|--------------------------|------------------------------------|-------------------------|
| `lidarToNumpy`           | `sensor_msgs/PointCloud2`          | `(N, F)` float32 array  |
| `compressedImageToNumpy` | `sensor_msgs/CompressedImage`      | `(H, W, 3)` uint8 array |
| `transformToMatrix`      | `geometry_msgs/Transform`          | `(4, 4)` float32 matrix |
| `interpolateMatrix`      | Two `(4, 4)` matrices + alpha      | `(4, 4)` float32 matrix |

---

## 2.0 `lidarToNumpy`

### 2.1 Background: PointCloud2 Memory Layout

#### 2.1.1 Structure

A `sensor_msgs/PointCloud2` message stores all points as a flat byte blob (`data`). Each point occupies exactly `point_step` bytes. Within that block, each channel (x, y, z, intensity, ring, etc.) sits at a specific byte position defined by its `PointField.offset`.

```
|<----------- point_step = 20 bytes ----------->|
[ x(4) | y(4) | z(4) | ring(2) | pad(2) | i(4) ]  ← point 0
[ x(4) | y(4) | z(4) | ring(2) | pad(2) | i(4) ]  ← point 1
...
```

`PointField` offsets for this layout:

| Field     | Offset | Datatype | Size |
|-----------|--------|----------|------|
| x         | 0      | float32  | 4 B  |
| y         | 4      | float32  | 4 B  |
| z         | 8      | float32  | 4 B  |
| ring      | 12     | uint16   | 2 B  |
| intensity | 16     | float32  | 4 B  |

#### 2.1.2 Ring

`ring` is the laser beam index on a multi-layer LiDAR (e.g. Velodyne VLP-16 has 16 beams, `ring=0` bottom, `ring=15` top). It identifies which horizontal scan line a point belongs to and is used for structured processing (ground segmentation, neighbour search). It is not extracted by default.

#### 2.1.3 Padding

Padding bytes are inserted by the driver to keep subsequent fields aligned to their own size (float32 must be at an address divisible by 4). They carry no data. In the layout above, 2 padding bytes after `ring` ensure `intensity` lands at offset 16 (divisible by 4).

### 2.2 Implementation

#### 2.2.1 Byte Buffer Parsing

The raw byte blob is interpreted as a `(N, point_step)` uint8 matrix — one row per point. This gives byte-level addressability so any field can be sliced out by its exact offset without making assumptions about what surrounds it.

```python
pointByteBuffer = np.frombuffer(lidarMsg.data, dtype=np.uint8).reshape(-1, lidarMsg.point_step)
fieldBytes = pointByteBuffer[:, field.offset:fieldEndByte].tobytes()
pointCloud[:, index] = np.frombuffer(fieldBytes, dtype=fieldDtype)
```

`np.frombuffer(..., dtype=np.uint8)` is a zero-copy view of the original data. The `.tobytes()` call is a necessary copy — the column slice is non-contiguous in memory (rows are `point_step` bytes apart) and `np.frombuffer` requires a contiguous buffer.

#### 2.2.2 Field Dtype Handling

`POINTFIELD_DATATYPE_MAP` maps the `PointField.datatype` integer constants (defined by the ROS2 `sensor_msgs` spec, not mcap) to numpy dtypes. Each field is read using its native dtype and cast to float32 on assignment into the output array. This correctly handles sensors where fields like `intensity` are uint16 rather than float32.

### 2.3 API Design

```python
def lidarToNumpy(lidarMsg, fieldNames=DEFAULT_LIDAR_MESSAGE_FIELDS)
```

Field names are a parameter rather than a hardcoded constant for two reasons:

1. **Different LiDAR configurations expose different fields.** Velodyne, Ouster, and Livox clouds all have different channel sets. Hardcoding `["x", "y", "z", "intensity"]` would require code changes to support other sensors.
2. **Output shape is derived from `fieldNames`.** The output is `(N, len(fieldNames))`, so callers can request any subset or ordering without modifying the function.

`DEFAULT_LIDAR_MESSAGE_FIELDS = ["x", "y", "z", "intensity"]` is used when no `fieldNames` argument is passed.

### 2.4 Dependency Decisions

#### 2.4.1 Why Not `sensor_msgs_py`

`sensor_msgs_py` provides `read_points()` which correctly handles offsets and `point_step`. It was considered and rejected because it is **not on PyPI** — it is distributed as a ROS2 apt package (`ros-<distro>-sensor-msgs-py`). Declaring it in `pyproject.toml` would produce a dependency that `pip` can never resolve, breaking installation for all users without a full ROS2 environment.

---

## 3.0 `compressedImageToNumpy`

*To be documented.*

---

## 4.0 `transformToMatrix`

*To be documented.*

---

## 5.0 `interpolateMatrix`

### 5.1 Background: Interpolating Between Two Rotations

In `synchronizer.py`, TF messages record the exact pose of a sensor at specific timestamps. A LiDAR scan arrives at a timestamp that falls between two TF readings, so the transform must be reconstructed by interpolation. The translation component (x, y, z) can be interpolated with a plain linear lerp. Rotation requires a different approach.

You can't average two orientations like scalar numbers — rotations don't work that way mathematically. Internally, rotations are represented as **quaternions**: 4-component unit vectors that live on the surface of a 4D unit sphere. The path between two rotations is an arc on that sphere. Two methods exist for traversing it:

#### 5.1.1 NLERP — Normalized Linear Interpolation

The naive approach: draw a straight line between the two quaternions through the interior of the sphere, find the point at `alpha` along that line, then project it back onto the surface by normalizing.

```
q = lerp(q1, q2, alpha)
q = q / |q|
```

The straight-line shortcut does not follow the arc — the interpolated rotation moves faster near the midpoint and slower near the endpoints. Angular velocity is not constant.

#### 5.1.2 SLERP — Spherical Linear Interpolation

The correct approach: walk along the surface of the sphere (the great-circle arc) using sin-weighted blending:

```
q = sin((1-α)·θ)/sin(θ) · q1  +  sin(α·θ)/sin(θ) · q2
```

where `θ` is the total angle between the two rotations. This gives **constant angular velocity** throughout the interpolation.

#### 5.1.3 Why SLERP for This Use Case

A real sensor rotates at physically constant angular velocity between measurements. SLERP matches this physical reality; NLERP introduces a timing distortion — the reconstructed pose at a given timestamp won't actually be where the sensor was at that moment. For a dataset used to train or evaluate a model, that's a systematic geometric error baked into every frame.

NLERP and SLERP converge when the angle between the two rotations is very small, so the difference is negligible when TFs are published at high frequency. But TF entries are cached over a rolling window (`TF_CACHE_SIZE`) and may bracket larger angular gaps, making SLERP the correct default.

### 5.2 Implementation

```python
def interpolateMatrix(startMatrix, endMatrix, alpha=0.5)
```

Translation and rotation are handled independently:

- **Translation:** plain linear lerp — `(1 - alpha) * t1 + alpha * t2`.
- **Rotation:** `scipy.spatial.transform.Slerp`, which also handles quaternion sign disambiguation (the fact that `q` and `-q` represent the same rotation but give the wrong shortest-path arc without a sign correction) internally.

`alpha` is the normalized time parameter in `[0, 1]`, computed in `synchronizer.py` as:

```python
alpha = (targetTimestamp - before[TIMESTAMP]) / (after[TIMESTAMP] - before[TIMESTAMP])
alpha = np.clip(alpha, 0.0, 1.0)
```

`np.clip` guards against floating-point edge cases where the two timestamps are nearly equal.

The default `alpha=0.5` is a convenience for direct or test usage and has no effect on production code, which always passes an explicitly computed alpha.

### 5.3 API Design

The function takes and returns full `(4, 4)` homogeneous matrices rather than separate translation/rotation components so callers never need to decompose or recompose a transform. The bottom row `[0, 0, 0, 1]` is preserved from `np.eye(4)` and never modified.

---

## 6.0 FAQ

**Q: Why is `np.uint8` hardcoded in the `np.frombuffer` call inside `lidarToNumpy`?**
It is not a configuration value — it is the correct dtype for byte-level addressing. Using `np.uint8` makes each array index correspond to exactly one byte, which is required for `field.offset` (measured in bytes) to index correctly. Any wider dtype would corrupt the offset arithmetic.

**Q: Why does `.tobytes()` create a copy? Can it be avoided?**
The column slice `pointByteBuffer[:, offset:end]` is non-contiguous in memory because rows are `point_step` bytes apart. `np.frombuffer` requires a flat contiguous buffer, so `.tobytes()` is necessary. The copy is per-field and bounded to `N × itemsize` bytes — acceptable for typical point cloud sizes.

**Q: Why is `POINTFIELD_DATATYPE_MAP` a module-level constant and not inside the method?**
It is the same for every call and every message. Defining it at module level avoids recreating the dict on each invocation and makes it available for testing or reuse elsewhere in the module.

**Q: What happens if `point_step` in the message is wrong?**
`reshape(-1, lidarMsg.point_step)` will raise a `ValueError` if `len(data)` is not divisible by `point_step`. This is the correct behaviour — a malformed message should not silently produce partial results.

**Q: Why is `ring` not in `DEFAULT_LIDAR_MESSAGE_FIELDS`?**
Ring is not universally present across all LiDAR models and is not needed for the core KITTI-format output (x, y, z, intensity). Callers that need it can pass a custom `fieldNames` list.
