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

*To be documented.*

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
