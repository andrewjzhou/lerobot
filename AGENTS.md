This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).

## Local Fork Notes: OpenArm v2 Support (July 2026)

This fork adapts the OpenArm Mini teleoperator and OpenArm follower to **OpenArm v2**
hardware (upstream code targets v1). Setup: bimanual OpenArm v2 followers (Damiao
motors, CAN FD via `can0`=right / `can1`=left) teleoperated by two OpenArm Mini
leaders (Feetech STS3215, USB serial).

### v2 hardware differences discovered (and patched)

- **Mirrored grippers**: on v2 the left gripper opens toward **+65°** and the right
  toward **-65°** (v1 opened both toward -65°). Patched:
  - `src/lerobot/teleoperators/openarm_mini/openarm_mini.py` — per-side gripper
    scale (`GRIPPER_TELEOP_TO_DEGREES_LEFT = +0.65`); squeeze = 0° = closed on both sides.
  - `src/lerobot/robots/openarm_follower/config_openarm_follower.py` —
    `LEFT_DEFAULT_JOINTS_LIMITS["gripper"] = (0.0, 65.0)`.
- **Wrist axes correspond directly** on v2: removed the v1 leader joint 6 ↔ follower
  joint 7 cross-remap (`JOINT_REMAP` now empty) and inverted joint_6's direction in
  `SIDE_MOTORS_TO_FLIP` on both sides.

### Hardware/calibration state (not in git)

- Left follower gripper motor was re-zeroed (Damiao set-zero, persists in motor
  flash) so **closed = 0°, open ≈ +76°** — the v2 stock convention matching the
  patched code. If that motor is ever re-zeroed, zero it at the fully closed jaw.
- Leader calibration files (`~/.cache/huggingface/lerobot/calibration/`) are stock
  output of `lerobot-calibrate`; no hand edits required by this fork.
- Note: the follower gripper motors respond on their ESC/MST CAN IDs but ignore the
  0x7FF param channel (refresh/param queries) — reads rely on MIT-command feedback,
  so occasional "Packet drop: gripper" warnings at startup are expected and benign.
- A gripper commanded outside its physical range stalls against its end stop and can
  sag the 24V rail until motors trip (blinking LED = latched fault; clear by power
  cycle). The patches above make that unreachable in normal teleop.
