# Multi-Modal Robotics Data Pipeline: MCAP to HDF5

## 1. Overview

The project implements a high-performance data engineering pipeline designed to convert raw multi-modal robotics telemetry stored in a **ROS2 MCAP** file into a **HDF5 dataset** format, optimized for deep learning. 

## 2. Architecture

The pipeline is built on a modular, decoupled architecture following the **Single Responsibility Principle**:

* **`reader.py` (Source):** A memory-efficient generator that streams raw ROS2 messages from MCAP files.
* **`synchronizer.py` (The Brain):** Manages message buffers and implements the logic for pairing LiDAR frames with the closest camera images.
* **`message_converter.py` (Service):** A stateless service class that handles binary-to-numpy conversion and **SLERP (Spherical Linear Interpolation)** for rotation matrices.
* **`hdf5_writer.py` (Sink):** Manages the HDF5 schema, dataset resizing, and persistent storage of sensor intrinsics and static transforms.



## 3. Data Schema

The HDF5 file is structured to support $O(1)$ random access, enabling high-speed training loops without pre-processing.

| Group | Dataset | Type | Description |
| :--- | :--- | :--- | :--- |
| `samples` | `timestamps` | `float64` | The reference timestamp (LiDAR-based) for the sample. |
| `camera` | `images` | `uint8` | Fixed-shape RGB tensors `(N, H, W, 3)`. |
| `lidar` | `data` | `float32` | A flat 1D array of all LiDAR points `(X, Y, Z, I)`. |
| `lidar` | `offsets` | `int64` | The starting index of a specific frame in the flat data array. |
| `lidar` | `counts` | `int32` | The number of points contained in that frame. |
| `transforms` | `[frame_id]` | `float32` | `(N, 4, 4)` Interpolated homogeneous matrices. |

## 4. Setup 

### 4.1 Folder Structure

```text
├── data
│   ├── processed/          # Generated HDF5 datasets
│   └── raw/                # Input kitti.mcap files
├── pipeline/               # Core Pipeline Package
│   ├── config.py           # Constants, Topic Names, and Thresholds
│   ├── hdf5_writer.py      # HDF5 I/O and Resizing logic
│   ├── __init__.py         # Module exposure
│   ├── message_converter.py# Math and Data transformation
│   ├── reader.py           # MCAP streaming generator
│   └── synchronizer.py     # Temporal state management
├── dataset.py              # PyTorch Dataset and DataLoader implementation
├── generate_dataset.py     # Pipeline Orchestrator (Main Entry)
├── requirements.txt        # Pinned project dependencies
└── README.md
```

### 4.2 Installing Dependencies

Recommended Python version: `3.10` or higher.

```bash
pip install -r requirements.txt
```

## 5. Running the pipeline

To generat ethe HDF5 dataset from your raw MCAP file, execute the dataset orchestrator: 

```bash 
python3 generate_dataset.py
```
To verify the data with the Pytorch DataLoader:

```bash
python3 dataset.py
```

