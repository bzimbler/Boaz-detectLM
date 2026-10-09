# Harbor framework overview

Harbor is a framework for evaluating and optimizing AI agents and language models
inside isolated sandboxes. It is the official harness for Terminal-Bench 2.0 and
ships integrations for many agents (Claude Code, OpenHands, Codex CLI, …) and
sandbox providers (Docker, Daytona, Modal, **Islo**, …).

Upstream: https://github.com/harbor-framework/harbor  
Docs: https://harborframework.com/docs  
Cookbook: https://github.com/harbor-framework/harbor-cookbook

## Install

```bash
uv tool install harbor
# or
pip install harbor
# Islo support:
pip install 'harbor[islo]'
```

## Core objects

| Object | Meaning |
|--------|---------|
| **Task** | Instruction + environment definition + verifier (`task.toml`, `instruction.md`, `environment/`, `tests/`) |
| **Dataset** | Named collection of tasks (`terminal-bench@2.0`) |
| **Agent** | Implements `BaseAgent` — runs inside the environment |
| **Environment** | Implements `BaseEnvironment` — Docker, Islo, Modal, … |
| **Trial** | One agent attempt at one task |
| **Job** | Product of agents × tasks × attempts, run concurrently |
| **Verifier** | Uploads tests, runs `test.sh`, reads reward files |
| **Trajectory** | ATIF agent history (`agent/trajectory.json`) |

## How `harbor run` executes

1. CLI (`src/harbor/cli/`) builds a `JobConfig` from flags and/or YAML.
2. `Job.create` resolves tasks (`TaskClient`), runs environment/agent preflight, expands trials.
3. `Trial.run` → `environment.start` → `agent.setup` / `agent.run` → artifact sync → `Verifier.verify` → `environment.stop`.
4. Results land under `jobs/` as `result.json` plus trajectories; inspect with `harbor view`.

## Task directory shape

```
my-task/
├── instruction.md
├── task.toml
├── environment/          # Dockerfile | docker-compose.yaml
├── tests/
│   └── test.sh           # writes /logs/verifier/reward.txt|json
└── solution/             # optional solve.sh for OracleAgent
```

## Selecting an environment

```bash
harbor run -e docker …          # default
harbor run -e islo …            # Islo microVM (needs harbor[islo] + ISLO_API_KEY)
harbor run -e daytona …
harbor run -e my_pkg:MyEnv …    # custom import path
```

Factory registry: `src/harbor/environments/factory.py` maps `EnvironmentType` to
lazy module paths and optional pip extras.

## Package map (`src/harbor/`)

See the interactive Architecture section in `../index.html`. High-signal packages:

- `cli/` — Typer commands (`run` ≡ `job start`)
- `trial/` — trial lifecycle
- `environments/` — providers including `islo.py`
- `agents/` — built-in and installed agents
- `verifier/` — reward collection
- `models/` — Pydantic configs and ATIF trajectories
