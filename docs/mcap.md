# MCAP File Format

## File Structure

MCAP is a binary container format developed by Foxglove for robotics sensor data. Every MCAP file follows this layout:

```
[Magic bytes]          \x89MCAP\r\n\r\n  (9 bytes, analogous to PNG magic)
[Header record]        file-level metadata (profile, library)
[Chunk]                compressed block: Schema + Channel + Message records
[Chunk]
...
[Chunk Index records]  one per chunk — offset, length, time range, per-channel message counts
[Statistics record]    file-wide summary: total message count, time range, per-channel counts
[Summary Offset]       points to where the summary section starts
[Footer]               points to summary and index sections; ends with magic bytes
```

### Key record types

| Record | Purpose |
|--------|---------|
| **Schema** | Embeds the full message type definition (e.g. ROS2 `.msg` source). One per message type. |
| **Channel** | Maps a topic string to a Schema and a serialization encoding. One per topic. |
| **Message** | The actual serialized payload, with a log timestamp and publish timestamp. Many per channel. |
| **Chunk** | Compressed container holding interleaved Schema, Channel, and Message records for a time range. |
| **Chunk Index** | Summary of one chunk: byte offset, time range, uncompressed size, per-channel count. |
| **Statistics** | File-wide aggregation: total messages, start/end time, per-channel message counts. |
| **Attachment** | Arbitrary binary blobs (e.g. calibration files, images). Not part of the message stream. |
| **Metadata** | Key-value string pairs for file-level annotations. |
| **Footer** | Byte offsets pointing to the summary section and summary offset record. Always at end of file. |

### Visual layout

```
┌─────────────────────────────────────────┐
│           MAGIC BYTES (9B)              │
│         \x89MCAP\r\n\r\n               │
├─────────────────────────────────────────┤
│              HEADER                     │
│         profile, library                │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐   │
│  │            CHUNK 1               │   │
│  │  ┌────────────────────────────┐  │   │
│  │  │ Schema  (sensor_msgs/Image)│  │   │
│  │  ├────────────────────────────┤  │   │
│  │  │ Channel (/camera/image)    │  │   │
│  │  ├────────────────────────────┤  │   │
│  │  │ Message  t=1.000s          │  │   │
│  │  │ Message  t=1.033s          │  │   │
│  │  │ Message  t=1.066s          │  │   │
│  │  ├────────────────────────────┤  │   │
│  │  │ Schema  (PointCloud2)      │  │   │
│  │  ├────────────────────────────┤  │   │
│  │  │ Channel (/lidar/points)    │  │   │
│  │  ├────────────────────────────┤  │   │
│  │  │ Message  t=1.000s          │  │   │
│  │  │ Message  t=1.100s          │  │   │
│  │  └────────────────────────────┘  │   │
│  │      compression: lz4            │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │            CHUNK 2               │   │
│  │        (next time window)        │   │
│  │            ...                   │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │            CHUNK N               │   │
│  │        (last time window)        │   │
│  │            ...                   │   │
│  └──────────────────────────────────┘   │
│                                         │
├─────────────────────────────────────────┤
│           CHUNK INDEX x N               │
│   byte_offset, time_range,              │
│   per_channel_count                     │
├─────────────────────────────────────────┤
│             STATISTICS                  │
│   total_messages: 12400                 │
│   start_time: 1.000s                    │
│   end_time:   62.300s                   │
│   channel_message_counts:               │
│     /camera/image  → 1860               │
│     /lidar/points  →  623               │
│     /tf            → 9917               │
├─────────────────────────────────────────┤
│          SUMMARY OFFSET                 │
├─────────────────────────────────────────┤
│              FOOTER                     │
│         MAGIC BYTES (9B)                │
└─────────────────────────────────────────┘
```

Every message lives inside a chunk — there are no loose messages at the file level. Schema and Channel records are stored inside chunks too, making each chunk independently decodable without reading the rest of the file.

### How reading works

- **Fast path** (`get_summary()`): seek to Footer → jump to Statistics and Chunk Index records. O(1) regardless of file size. This is how `mcap2hdf5 inspect` gets topic lists and message counts without scanning the file.
- **Slow path** (`iter_messages()`): scan every Chunk sequentially, decompress, yield records. Needed only when the Statistics block is absent (malformed or truncated file).
- **Random access**: Chunk Index records give the byte offset and time range of each chunk, so a reader can seek directly to any time window without touching the rest of the file.

---

## Compression

Chunks are individually compressed. Supported codecs: `lz4`, `zstd`, `none`. Each chunk can use a different codec. This means:

- A file can mix compressed and uncompressed chunks (e.g. uncompressed for fast scrubbing, compressed for archival).
- Decompression is parallelisable — chunks are independent.
- Random seeking within a compressed file is practical: you only decompress the chunks you need.

---

## Self-describing format

Schema records are embedded in the file itself. You do not need ROS, a message registry, or any external type system to deserialise an MCAP file — everything needed to interpret the bytes is inside the file. This is a deliberate design choice that makes MCAP viable outside the ROS ecosystem.

Supported serialization encodings include: `ros2`, `ros1`, `protobuf`, `json`, `flatbuffers`. The encoding is declared per Channel record, so a single MCAP file can mix encodings across topics.

---

## Comparison: MCAP vs ROS2 .db3

The `.db3` format was the default ROS2 bag format from ROS2 Foxy through Galactic. MCAP replaced it as the default in **ROS2 Humble (May 2022)**.

| Property | MCAP | ROS2 `.db3` (SQLite) |
|----------|------|----------------------|
| Underlying format | Custom binary container | SQLite3 database |
| Compression | Per-chunk (lz4, zstd) | None |
| Random access | Via Chunk Index (byte offset + time range) | SQL `WHERE timestamp BETWEEN` |
| Inspection without full scan | Yes — Statistics record | No — requires `COUNT(*)` query |
| Self-describing | Yes — Schemas embedded in file | No — requires ROS type system |
| Serialization formats | ROS1, ROS2, Protobuf, JSON, FlatBuffers | ROS2 CDR only |
| Multi-topic splitting | Single file, all topics | Single file, all topics |
| Tooling ecosystem | Foxglove Studio, mcap CLI, Python/C++/Go/Rust SDKs | rosbag2 CLI, limited third-party |
| File portability | High — readable without ROS installed | Low — needs ROS2 and correct message packages |
| Large file performance | Good — chunked layout, parallel decompression | Degrades — SQLite page cache pressure |

---

## Why MCAP matters for ML pipelines

- **No ROS dependency at inference time.** You can read MCAP files in a pure Python environment (`pip install mcap`) without installing ROS2. This is critical for training infrastructure that shouldn't carry a ROS dependency.
- **Inspection is free.** The Statistics record means you can audit a recording (topic list, message counts, duration) in milliseconds. For a dataset pipeline this means fast validation before committing to a full conversion run.
- **Streaming is natural.** The chunked layout maps directly onto a streaming ETL pattern: decompress one chunk, process it, move on. Memory usage is bounded by chunk size, not file size.
- **Schema evolution is traceable.** Because Schema records are embedded, you can tell exactly what message definition was used to record a file — not what definition is currently installed on your machine. This matters when replaying old bags after upgrading message packages.

---

## Interview talking points

**Why is MCAP better than a bag of CSVs or raw numpy files?**
MCAP preserves the original message schemas, timestamps (both publish and receive), and topic structure. It is the authoritative source of truth for what the robot observed. Downstream formats like HDF5 are derived — optimised for ML access patterns but not for re-processing or debugging.

**What happens when `get_summary()` returns None?**
The file has no Statistics record — either it was truncated during recording (e.g. a crash mid-write), or it was written by a tool that skipped the summary. You fall back to `iter_messages()` which scans every chunk. For large files this is expensive. In `mcap2hdf5` this triggers `countMessagesByChannel()` with a progress bar.

**How does MCAP handle time synchronisation across topics?**
It doesn't — that is intentional. MCAP records each message with a log timestamp (when the recorder received it) and an optional header timestamp (when the sensor produced it). Multi-topic time alignment is the responsibility of the consumer. This is why `SensorDataSynchronizer` exists in mcap2hdf5: MCAP gives you the raw timestamped stream; synchronisation is a pipeline concern.

**What is the difference between log time and publish time?**
`log_time` is when the MCAP recorder wrote the message (wall clock of the recording machine). `publish_time` is the timestamp in the message header, set by the sensor driver or node that published it. For high-frequency sensors these can diverge significantly under CPU load. ML pipelines should use `publish_time` (header timestamp) for synchronisation, not `log_time`.

**Why are Chunk Index records useful beyond random access?**
They give you per-channel message counts and time ranges per chunk without decompressing anything. You can determine the temporal coverage of each topic across the file in one pass over the index — useful for detecting dropped topics or gaps in recording.

---

## Schemas, Channels, and Messages — the lookup chain

Every message in an MCAP file is decoded by following a two-step chain:

```
Message ──(channel_id)──▶ Channel ──(schema_id)──▶ Schema
  │                          │                         │
raw bytes              topic name              type definition
log_time               encoding                (.msg source)
publish_time
```

Example for a camera topic:

```
Schema  id=1  name="sensor_msgs/msg/Image"
              data="std_msgs/Header header
                    uint32 height
                    uint32 width
                    string encoding
                    uint8[] data"

Channel id=7  topic="/camera/image"
              schema_id=1
              message_encoding="cdr"

Message       channel_id=7
              log_time=1700000000123456789
              publish_time=1700000000120000000
              data=<raw CDR bytes>
```

If two topics share the same message type (e.g. `/camera/left/image` and `/camera/right/image`), they have two Channel records but both point to the same Schema — the type definition is stored once.

In `inspectMcap`, `summary.schemas.get(channel.schema_id)` walks the Channel → Schema arrow to get the human-readable type name for the topic table display.

---

## Chunking in detail

### What is a chunk

A chunk is a compressed, self-contained time window of messages. It holds interleaved Schema, Channel, and Message records for all topics that received messages during that window. Every message in an MCAP file lives inside a chunk — there are no loose messages at the file level.

### How the recorder decides chunk boundaries

The recorder does not use sensor timestamps to decide when to close a chunk. It uses two simple triggers based on `log_time` (wall clock of the recording machine):

```
message arrives → write to current open chunk
                        ↓
          chunk size ≥ threshold?   (default ~1MB)
          OR wall-clock timeout?
                        ↓
              close chunk → compress → write to file
              open new chunk
```

This means a single chunk will contain interleaved messages from all active topics during that wall-clock window, regardless of their sensor timestamps. There is no guarantee that chunk boundaries align with any meaningful sensor event.

### Configuring chunking at record time

```bash
ros2 bag record -a --storage mcap --storage-config-file storage_config.yaml
```

```yaml
# storage_config.yaml
compression_mode: message
compression_format: zstd
chunk_size: 4194304        # 4MB in bytes
```

Or via the Python writer directly:

```python
from mcap.writer import Writer

with open("output.mcap", "wb") as f:
    writer = Writer(f, chunk_size=4 * 1024 * 1024)
```

Larger chunks → better compression ratio, slower random access.
Smaller chunks → faster seeking, worse compression ratio.

### Chunk boundaries vs sensor gaps

`MAX_CHUNK_GAP` in `SensorDataSynchronizer` is not the same as a chunk boundary. It detects gaps in a sensor's `publish_time` sequence — a signal that the sensor paused or the recording split into a new sequence. A recording can have many chunk boundaries within one continuous sensor sequence, or a sensor gap that spans multiple chunks. They are independent concepts.

---

## Magic bytes

Magic bytes are a fixed sequence at the start (and end) of a file that identify the format before any parsing begins.

MCAP magic: `\x89 M C A P \r \n \r \n`

| Byte | Purpose |
|------|---------|
| `\x89` | Non-ASCII — causes immediate rejection by ASCII-only systems (old email relays, FTP text mode) |
| `MCAP` | Human-readable format identifier |
| `\r\n\r\n` | Line-ending probe — if a tool corrupts `\r\n` to `\n` during transfer, the magic won't match and you know the file is damaged |

Other formats follow the same pattern:
```
PNG:  \x89 P N G \r \n \x1a \n
PDF:  % P D F -
ZIP:  P K \x03 \x04
ELF:  \x7f E L F
```

MCAP places magic bytes at both the start and end of the file. This means a reader can verify the file was fully written — if the recorder crashed before the Footer was written, the closing magic bytes are absent and the file is known to be incomplete. This is also why `get_summary()` can return `None`: no Footer means no magic at the end means the file was truncated mid-recording.

The `file` command on Linux uses magic bytes (not file extensions) to identify formats. It checks against a database of known sequences — MCAP is in that database.

---

## Is MCAP a relational database?

No — it shares surface similarities but the design goals are opposite.

**What looks relational:**
- Schema records define types (like table schemas)
- Channel records reference Schemas by ID (like a foreign key)
- Message records reference Channels by ID (like rows referencing a table)

**Why it isn't:**

| Property | Relational DB | MCAP |
|----------|---------------|------|
| Write pattern | Random read/write/update | Append-only, write once |
| Query | SQL — arbitrary predicates | Seek by time range only |
| Constraints | Schema enforced at write | Schema is documentation only — not validated |
| Physical order | By primary key or heap | By `log_time` arrival order |
| Updates/deletes | Supported | Impossible — data is in compressed chunks |
| Transactions | Full ACID | None — file is only valid after Footer is written |

The closer analogy is a **write-ahead log (WAL)** — like the one Postgres uses internally for crash recovery. Append-only, sequential, time-ordered, with an index at the end for fast lookup. Or even simpler: a structured binary log file with an index, not a database.
