# Autonomous Space Robotics — Code

Course code for Fall 2026 Autonomous Space Robotics.

## Setup

Install [uv](https://docs.astral.sh/uv/) (once):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from the repository root:

```bash
uv sync
```

This creates a `.venv/` and installs the pinned dependencies from `uv.lock`. uv downloads
a suitable Python automatically — you do not need one installed already.

## Running

Run scripts with `uv run`; there is no need to activate the virtualenv.

```bash
cd lecture1
uv run python spacecraft_sim.py   # rigid-body spacecraft in LEO -> spacecraft_sim.h5 + 3D plot
uv run python test_mujoco.py      # MuJoCo pendulum in the interactive viewer
```
