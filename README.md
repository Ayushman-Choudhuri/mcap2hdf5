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

## Overview

mcap2hdf5 is a streamlined tool for transforming ROS2 MCAP sensor recordings into synchronized, ML-ready HDF5 datasets.

> **Supported modalities for upcoming 0.1.0 release:** camera, LiDAR, transforms

---

## Installation

Python 3.10+ required.

```bash
pip install git+https://github.com/Ayushman-Choudhuri/mcap2hdf5.git
```

> PyPI release planned for v0.1.0.

---

## Quick Start

```bash
# 1. Inspect an MCAP file — topic table and auto-detection
mcap2hdf5 inspect data.mcap

# 2. Generate a job config YAML
mcap2hdf5 init data.mcap

# 3. Edit the generated YAML if needed, then convert
mcap2hdf5 convert data_config.yaml
```

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/job_config.md](docs/job_config.md) | Full job config YAML reference and tuning guide |
| [docs/hdf5.md](docs/hdf5.md) | HDF5 output schema, flat LiDAR pool pattern, PyTorch DataLoader |
| [docs/mcap.md](docs/mcap.md) | MCAP file format internals |

---

## Roadmap

| Version | Scope |
|---------|-------|
| **0.1.0** | Single LiDAR + single camera. `inspect`, `init`, `convert` CLI. Auto-detect topics. |
| **0.2.0** | Multiple cameras and LiDARs. Named subgroup schema. |
| **1.0.0+** | IMU, Radar, GPS, Odometry. |

---

## License

MIT
