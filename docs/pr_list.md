# PR List — mcap2hdf5 V1

PRs are ordered by dependency. Each builds on the last. Do not reorder unless the dependency chain permits it.

---

## Phase 1 — Repo Foundation

### PR-01: Restructure repo into installable Python package

Move `pipeline/` into a proper `mcap2hdf5/` package directory so the tool can be installed with `pip`.

- Create `mcap2hdf5/` directory and move all files from `pipeline/` into it
- Fix the typo: rename `pipeline/__init__,py` → `mcap2hdf5/__init__.py`
- Add `pyproject.toml` with build system (`hatchling`), all existing dependencies, and the entry point `mcap2hdf5 = "mcap2hdf5.cli:app"` (the cli module doesn't exist yet — leave the entry point defined but pointing at a not-yet-created file is fine for now)
- Update all imports in `generate_dataset.py` and `dataset.py` to use `mcap2hdf5.*`
- Verify `pip install -e .` works and `python3 generate_dataset.py` still runs without errors

---

### PR-02: Add CI with ruff linting and pytest

Establish the quality gate before any new code is written.

- Add `.github/workflows/ci.yaml`: trigger on push and PR to `main`; run `ruff check .` and `pytest` across Python 3.10, 3.11, 3.12 using a matrix strategy
- Add ruff config in `pyproject.toml` under `[tool.ruff]`: enable `E`, `F`, `I` rule sets; set line length to 100
- Add `pytest` and `ruff` to `[project.optional-dependencies]` under a `dev` extra
- There are no tests yet — `pytest` should exit 0 with "no tests collected"

---

## Phase 2 — Config Layer

### PR-03: Implement PipelineConfig with YAML parsing and validation

Replace the hardcoded constants in `pipeline/config.py` with a proper typed config loaded from a YAML file.

- Create `mcap2hdf5/config.py` with dataclasses: `LiDARSensorConfig(name, topic, fields)`, `CameraSensorConfig(name, topic, info_topic)`, `SyncConfig(reference, threshold_ms, chunk_gap_ms, tf_cache_size)`, `OutputConfig(compression, write_batch_size, initial_lidar_capacity)`, and `PipelineConfig` as the root
- Write `loadConfig(path: Path) -> PipelineConfig` that reads YAML, fills defaults for optional fields, and raises `ValueError` with a human-readable message on missing required fields or invalid values
- Keep the old `pipeline/config.py` constants alive in `mcap2hdf5/constants.py` with a deprecation comment — they are still used by the existing code and will be replaced incrementally in later PRs
- Add `pyyaml` to dependencies in `pyproject.toml`
- Add `tests/test_config.py`: test valid config loads correctly, test missing `sensors` raises, test defaults are applied

---

## Phase 3 — CLI Skeleton

### PR-04: Add Typer CLI with stub commands

Give the package its CLI entry point before any command logic exists.

- Create `mcap2hdf5/cli.py` with a `typer.Typer()` app
- Add three commands: `inspect(mcap_file: Path)`, `init(mcap_file: Path, output: Path)`, `convert(mcap_file: Path, config: Path, output: Path)` — each body raises `typer.Exit()` with a "not yet implemented" message printed via `typer.echo`
- Add `typer` to dependencies in `pyproject.toml`
- Verify: after `pip install -e .`, running `mcap2hdf5 --help` shows all three commands with correct argument names; `mcap2hdf5 convert --help` shows `--config` and `--output` flags

---

## Phase 4 — `inspect` Command

### PR-05: Add MCAPSource.listTopics() — index-only topic scan

Read topic statistics from an MCAP without streaming every message payload.

- Add `listTopics(mcapPath: Path) -> dict[str, TopicInfo]` as a standalone function in `mcap2hdf5/reader.py` (not a method on `MCAPSource`)
- `TopicInfo` is a dataclass: `schema_name: str`, `message_count: int`, `start_time: float`, `end_time: float`
- Use the MCAP index (channel and statistics records) via the `mcap` library directly — do not use `read_ros2_messages` which streams everything; use `mcap.reader.McapReader` and read `statistics` and `channels` from the summary
- Add `tests/test_reader.py` with a test against a real or fixture MCAP that verifies topic names and message counts are correct

---

### PR-06: Implement `inspect` CLI command

Display the topic table to the user.

- Implement the `inspect` command body in `cli.py`: call `listTopics()`, then print a formatted table
- Add `rich` to dependencies; use `rich.table.Table` for output — columns: Topic, Type, Messages, Duration (end − start in seconds)
- Print recording total duration and message count as a header above the table
- If the file does not exist, print a clear error and exit with code 1

---

## Phase 5 — `init` Command

### PR-07: Implement modality auto-detection from ROS2 message types

Map raw topic info to sensor candidates without any user input.

- Create `mcap2hdf5/detect.py`
- Write `detectSensors(topics: dict[str, TopicInfo]) -> DetectedSensors` where `DetectedSensors` holds lists of `DetectedLiDAR` and `DetectedCamera`
- Rules: `sensor_msgs/msg/PointCloud2` → lidar; `sensor_msgs/msg/CompressedImage` or `sensor_msgs/msg/Image` → camera; `sensor_msgs/msg/CameraInfo` → pair to camera by longest matching topic prefix
- Derive `name` from the topic path: strip common prefixes like `/sensor/`, `/sensors/`, take the penultimate path segment (e.g. `/sensor/camera/left/image_raw/compressed` → `left`)
- Add `tests/test_detect.py`: test with a dict of fake `TopicInfo` entries, verify correct sensor detection and CameraInfo pairing

---

### PR-08: Implement YAML config template generation

Turn detected sensors into a config file the user can edit.

- Add `generateConfigTemplate(detected: DetectedSensors) -> str` in `detect.py`
- Output is a YAML string with all detected LiDAR and camera sensors filled in, default sync parameters, and default output settings
- Topics with unsupported types (Imu, Radar, etc.) appear as YAML comments at the bottom: `# /imu/data (sensor_msgs/msg/Imu) — IMU support coming in v2`
- If no reference sensor can be determined (no LiDAR detected), leave `sync.reference` as `# TODO: set to your primary lidar name`
- Add `tests/test_detect.py`: parse the generated YAML string back with `loadConfig()` and verify it succeeds

---

### PR-09: Wire `init` CLI command

Connect detection and template generation to the command.

- Implement the `init` command body in `cli.py`: `listTopics()` → `detectSensors()` → `generateConfigTemplate()` → write to `--output` path if provided, otherwise print to stdout
- Print a summary before the YAML output: `Detected 2 LiDAR sensor(s), 2 camera sensor(s).`
- If `--output` is given and the file already exists, prompt for confirmation before overwriting using `typer.confirm()`

---

## Phase 6 — Modality Refactor (no behavior change)

### PR-10: Extract TF cache and SLERP interpolation into transforms.py

Isolate the TF logic so it can be unit-tested and reused independently of the synchronizer.

- Create `mcap2hdf5/transforms.py`
- Move `transformToMatrix()` and `interpolateMatrix()` out of `message_converter.py` into `transforms.py`
- Create a `TFCache` class that owns the rolling list per key, the cache size cap, and the `interpolate(key, timestamp) -> np.ndarray | None` method — absorbing the logic currently spread across `synchronizer.py`
- Update all imports in `synchronizer.py` and `message_converter.py` — zero behavior change
- Add `tests/test_transforms.py`: test `interpolateMatrix` at alpha=0, alpha=1, alpha=0.5 with known rotation quaternions; test SLERP sign correction (dot product flip)

---

### PR-11: Create LiDARConverter in modalities/lidar.py

Encapsulate PointCloud2 decoding behind a clean interface.

- Create `mcap2hdf5/modalities/base.py` with `ModalityConverter(ABC)` and one abstract method `toNumpy(self, rosMsg) -> np.ndarray`
- Create `mcap2hdf5/modalities/lidar.py` with `LiDARConverter(ModalityConverter)`; constructor takes `fields: list[str]`; `toNumpy` extracts those fields from the PointCloud2 binary buffer using the existing dtype-map logic from `message_converter.py`; raises `ValueError` if a required field is absent from the message
- Remove the LiDAR decoding code from `message_converter.py` and call `LiDARConverter` from there to keep existing callers working during the transition
- Add `tests/test_modalities.py`: construct a minimal synthetic PointCloud2 message in-memory with known x/y/z/intensity values, call `toNumpy`, verify output shape and values

---

### PR-12: Create CameraConverter in modalities/camera.py

Encapsulate image decoding for both compressed and raw image messages.

- Create `mcap2hdf5/modalities/camera.py` with `CameraConverter(ModalityConverter)`; no constructor arguments
- `toNumpy` detects whether the message is `CompressedImage` (has `.format` field) or `Image` (has `.encoding` field) and decodes accordingly — OpenCV for compressed, numpy reshape for raw
- Remove image decoding from `message_converter.py` and delegate to `CameraConverter`
- Add tests in `test_modalities.py`: test with a synthetic JPEG-encoded `CompressedImage` (encode a known numpy array via OpenCV, wrap in a fake message object, decode and compare)

---

## Phase 7 — Generalize Reader

### PR-13: Parameterize MCAPSource to accept dynamic info_topics

Remove the hardcoded topic names for metadata capture.

- Change `MCAPSource.__init__` to accept `infoTopics: set[str]` — the set of topics on which to capture the first-seen message as metadata
- Replace `getCameraMetadata()` and `getStaticTransforms()` with a single `getMetadata(topic: str) -> Any | None`
- Remove the two hardcoded topic constants (`CAMERA_INTRINSIC_PARAMETERS_TOPIC`, `TF_STATIC_TOPIC`) from the reader
- Update `generate_dataset.py` to pass the hardcoded topics explicitly so it continues to work during transition
- No behavioral change — existing functionality must still work

---

## Phase 8 — Generalize Synchronizer

### PR-14: Generalize SensorDataSynchronizer to N sensors

Remove all hardcoded topic and sensor constants from the synchronizer.

- Rewrite `SensorDataSynchronizer.__init__` to accept a `PipelineConfig`; build per-sensor buffers and `lastTimestamps` dicts from `config.sensors.lidar` and `config.sensors.camera` keyed by sensor name
- The reference sensor is identified by `config.sync.reference` (format `lidar/front`); its frames are used as the temporal anchor in `flushSamples()`
- Replace `SyncGroup` dataclass: `timestamp: float`, `sensorData: dict[str, dict]` (maps sensor name to its entry), `transforms: dict[str, np.ndarray]`
- Replace the single `TFCache` usage with `TFCache` from `transforms.py`
- Chunk gap detection runs independently per sensor topic using `config.sync.chunk_gap_ms`
- Remove all references to `LIDAR_TOPIC`, `CAMERA_IMAGE_TOPIC` constants
- Add `tests/test_synchronizer.py`: test with 1 LiDAR + 2 cameras using fake `StreamMessage` objects; verify pairing, verify frames outside threshold are dropped, verify gap detection triggers a flush

---

## Phase 9 — Generalize Writer

### PR-15: Generalize HDF5Writer for multiple LiDAR sensors

Each LiDAR sensor gets its own flat point pool.

- Rewrite `HDF5Writer.createDatasets` to iterate over `config.sensors.lidar` and create `lidar/{name}/data`, `lidar/{name}/offsets`, `lidar/{name}/counts` for each
- Each sensor's pool is initialized to `config.output.initial_lidar_capacity` and doubles independently on overflow
- `writeBatch` receives `SyncGroup` objects; calls `LiDARConverter(sensor.fields).toNumpy(...)` per sensor per sample
- Add `tests/test_writer.py`: write 10 samples with 2 LiDAR sensors of different field counts, read back and verify offset/count indexing returns correct points for each sensor

---

### PR-16: Generalize HDF5Writer for multiple camera sensors

Each camera gets its own HDF5 group with intrinsics stored as group attributes.

- `createDatasets` iterates over `config.sensors.camera` and creates `camera/{name}/images` dataset for each, with LZF compression
- In `finalize()`, for each camera that has captured metadata (via `MCAPSource.getMetadata(info_topic)`), write `k`, `d`, `r`, `p`, `distortion_model`, `width`, `height` as attributes on the `camera/{name}/` group — not as flat file attributes
- Remove the old flat file attributes `camera_k`, `camera_d`, etc.
- Add tests: write 2 cameras with different resolutions and intrinsics, read back and verify each camera's group attributes are correct and independent

---

### PR-17: Write file-level provenance attributes in HDF5Writer

Record how and when the file was created.

- In `finalize()`, write these attributes at the file root: `mcap2hdf5_version` (read from package `__version__`), `created_at` (UTC ISO 8601 string), `source_file` (basename of the input MCAP path), `num_samples`
- Add `__version__ = "0.1.0"` to `mcap2hdf5/__init__.py`
- Add a test: call finalize on a minimal writer, open the HDF5 file and assert all four attributes are present with correct types

---

## Phase 10 — Wire `convert` Command

### PR-18: Implement `convert` command — basic end-to-end

Replace `generate_dataset.py` with the proper CLI command.

- Implement the `convert` command body in `cli.py`: load config → construct `MCAPSource` with `infoTopics` from config → `SensorDataSynchronizer(config)` → `HDF5Writer(config, outputPath)` → stream, sync, write batches, finalize
- Use `logging.INFO` for per-batch messages: `Written batch of N samples (chunk M)`
- Delete `generate_dataset.py` — its logic now lives in the `convert` command
- Verify end-to-end on the existing KITTI MCAP with a manually written config YAML

---

### PR-19: Add per-chunk progress output to `convert`

Show the user something is happening for long-running recordings.

- After each `writeBatch` call, print a single line: `Chunk N | +M samples | total S | lidar/front: X pts`
- Use `rich.progress.Progress` with a spinner — no bar since total chunk count is unknown until the stream ends
- Keep logging statements for the file log and use `rich` only for the terminal output

---

### PR-20: Add completion summary table to `convert`

Print a per-sensor breakdown when conversion finishes.

- After `finalize()`, print a `rich.table.Table` with columns: Modality, Sensor, Samples, Notes
- Notes column: for LiDAR, show total point count formatted with commas; for camera, show `W×H, {compression}`
- If a sensor produced 0 samples (all frames dropped by the sync threshold), print a warning row in yellow: `camera/right — 0 samples (all frames exceeded sync threshold)`

---

## Phase 11 — Tests and Release

### PR-21: Create synthetic MCAP test fixtures

Provide small self-contained MCAP files for CI so tests never depend on real recordings.

- Write `tests/create_fixtures.py` — a standalone script (not a test) using the `mcap` writer API to produce:
  - `tests/fixtures/single_sensor.mcap`: 20 LiDAR frames + 20 CompressedImage frames on one camera, 10 Hz, 2 seconds, 100 points per frame
  - `tests/fixtures/multi_sensor.mcap`: 2 LiDAR topics + 2 camera topics, same duration
- Run the script once, commit the `.mcap` binaries to the repo
- Add a `tests/conftest.py` that provides `single_sensor_mcap` and `multi_sensor_mcap` as pytest fixtures returning the fixture file paths

---

### PR-22: Add integration test for full convert pipeline

Verify the whole pipeline produces a correct HDF5 file.

- Add `tests/test_integration.py`
- Test 1: run `convert` on `single_sensor.mcap` with a programmatically built `PipelineConfig`, open the output HDF5 and assert: `samples/timestamps` has length 20, `lidar/front/data` shape is `(2000, 4)`, `camera/left/images` shape is `(20, H, W, 3)`, `lidar/front/offsets[5]` equals 500, file attrs include `mcap2hdf5_version`
- Test 2: run on `multi_sensor.mcap` and verify both LiDAR groups and both camera groups exist with independent schemas

---

### PR-23: Add LICENSE, CONTRIBUTING.md, and README

Documentation needed before any public announcement.

- Add `LICENSE` — MIT license with the current year and your name
- Add `CONTRIBUTING.md`: dev setup (`pip install -e ".[dev]"`), how to run tests, how to add a new modality (subclass `ModalityConverter`, add to `modalities/`, update `detect.py` and `writer.py`), PR expectations
- Add `README.md`: one-paragraph description, install command, the three-command workflow (inspect → init → convert), full config YAML reference, full HDF5 schema reference, link to CONTRIBUTING

---

### PR-24: Add PyPI release workflow

Publish the package on a git tag push.

- Add `CHANGELOG.md` with a `v0.1.0` entry listing what's included
- Add `.github/workflows/release.yaml`: trigger on `v*` tag push; build with `hatch build`; publish to PyPI using `pypa/gh-action-pypi-publish` with a PyPI trusted publisher (OIDC, no token stored in secrets)
- Update `pyproject.toml`: set `version = "0.1.0"`, add `classifiers` (Development Status, Intended Audience, License, Programming Language), add `[project.urls]` for Homepage and Issues
