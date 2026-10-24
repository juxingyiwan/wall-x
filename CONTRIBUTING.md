# Contributing to Wall-X

[中文](CONTRIBUTING.zh-CN.md) · [Home](README.md)

Follow the [official contribution principles](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/CONTRIBUTING.md): focused changes, reproduction steps, documentation updates and no private paths/credentials or large checkpoints. Use an independent development environment.

<a id="checks"></a>

## 1. Run the relevant checks

From the repository root, Python 3.10 or 3.12 can run the core CPU regression environment used by CI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'pip>=25,<27'
python -m pip install --no-deps torch==2.10.0+cpu torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tests/requirements-cpu.txt
python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD:$PWD/deployment/client/src" python -m pytest tests -q
```

For deployment/client changes, install their development extras in a separate environment and run:

```bash
python -m pip install -e 'deployment/client[dev,vision]'
python -m pytest -c deployment/pyproject.toml deployment/tests -q
python -m pytest -c deployment/client/pyproject.toml deployment/client/tests -q
python scripts/check_docs.py
python scripts/check_robotwin_docs.py
python -m pip install pre-commit
pre-commit run --all-files
```

Use `.github/workflows/core-tests.yml`, `deployment.yml` and `lint.yml` as the executable check definitions, including Ruff and independent client wheel checks. Install a wheel outside the source directory when changing packaging. For an optional allocated eight-GPU regression:

```bash
PYTHONPATH="$PWD:$PWD/tests" torchrun --standalone --nproc_per_node=8 tests/b601_training_distributed.py /path/to/new-test-output
```

<a id="validation"></a>

## 2. Match validation to the change

| Change | Additional evidence |
| --- | --- |
| Data, padding, normalization or model inputs | Compare fixed recorded inputs and real-weight outputs with the baseline; state hashes/tolerances |
| Training, split or recovery | Short training plus checkpoint continuation with the same inputs/configuration |
| Simulator | Real-checkpoint rollout with pinned simulator/data/assets and recorded seeds |
| Robot lifecycle or hardware | CPU fault-injection tests, replay/preflight, then a separately attended trial |
| Documentation | Internal links, bilingual commands/structure, CLI parsing and shell syntax; no device access |

Imports, setup and CI must not open cameras/CAN or move motors. Preserve single-controller ownership, shared locks, units, stale-response rejection and shutdown cleanup.

<a id="documentation"></a>

## 3. Maintain the small documentation set

Maintain five English/Chinese document pairs: the root README, LIBERO/RoboTwin/ReBot user guides and this contribution guide. Keep sections, commands and parameters aligned between languages. Put complete usage instructions in the scenario guide, implementation notes in the [technical chapter at the end of the README](README.md#development), and validation summaries in the PR. Preserve the official upstream documents verbatim.

<a id="pull-requests"></a>

## 4. Prepare a focused PR

Describe the problem, final behavior and compatibility impact. Identify the candidate and official upstream baseline, explain shared-core changes, and include validation commands and results. Keep independent scenario changes reviewable separately.

For model, simulator or robot experiments, record code and resource versions, configuration, commands, seeds and results. Preserve copyright headers and `NOTICE.txt`. Keep private machine configuration, credentials, checkpoints, datasets and experiment logs outside the public source tree. Place selected documentation images and demo videos in `assets/` and include them in the package manifest.
