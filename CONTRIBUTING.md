# Contributing

Python 3.10+ required. This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies and the virtual environment.

## Setup

### Linux and macOS

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

Native Windows is not supported due to system-level dependencies required by `opencv-python-headless`. Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu 22.04, then follow the Linux instructions above.

Clone the repo and sync dependencies:

```bash
uv sync --extra dev 
```

## Running commands

```bash
uv run mcap2hdf5 --inspect data.mcap
uv run pytest
uv run ruff check .
uv run ruff check --fix .
```

Or activate the virtual environment manually and run commands directly:

```bash
source .venv/bin/activate
```
