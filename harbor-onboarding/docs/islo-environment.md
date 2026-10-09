# How `IsloEnvironment` is laid out

Detailed walkthrough of the Islo Harbor environment provider. Canonical source:

- Upstream / fork: `src/harbor/environments/islo.py`
- Standalone plugin: `harbor-env` → `src/harbor_islo/environment.py`

This document mirrors the style of islo-web-api’s `docs/code-structure.md`:
what each piece owns, which way calls may point, and the invariants that keep
trials safe.

## Repos

| Repo | Role |
|------|------|
| [harbor-framework/harbor](https://github.com/harbor-framework/harbor) | Upstream; ships `type: islo` |
| [islo-labs/harbor-fork](https://github.com/islo-labs/harbor-fork) | **Islo’s GitHub fork** of Harbor (`upstream` → harbor-framework) |
| [islo-labs/harbor-env](https://github.com/islo-labs/harbor-env) | PyPI plugin `harbor-islo` for a separate release train |
| [islo-labs/harbor](https://github.com/islo-labs/harbor) | Early copy; not the active fork |

## Class graph

```
GatewayRuleConfig(BaseModel)
GatewayConfig(BaseModel)
_IsloComposeOps(DinDComposeOps)          # compose transport adapter
IsloEnvironment(ComposeServiceOpsMixin, BaseEnvironment)
```

Factory entry (`environments/factory.py`):

```
EnvironmentType.ISLO → harbor.environments.islo:IsloEnvironment  (extra: islo)
```

## Constructor contract

```python
IsloEnvironment(
    gateway_profile: str | None = None,
    gateway: GatewayConfig | dict | None = None,
    delete_after_seconds: int = 3600,
    **kwargs,  # BaseEnvironment fields
)
```

Invariants enforced in `__init__`:

1. **XOR gateways** — `gateway_profile` and `gateway` cannot both be set.
2. **Env auth** — reads `ISLO_API_KEY`, `ISLO_API_URL` (default `https://api.islo.dev`), `ISLO_COMPUTE_URL`.
3. **Compose detection before `super()`** — `_validate_definition` depends on knowing whether `environment/docker-compose.yaml` (or `extra_docker_compose`) exists.
4. **Policy vs custom gateway** — if Harbor requested `allowlist` / `no-network`, a custom gateway is rejected (Harbor cannot prove the named profile matches the policy).
5. **Auto translation** — when no custom gateway is supplied, `_network_policy_gateway_config` is built from the task’s `NetworkPolicy`.
6. **`delete_after_seconds`** — must be `0` or a multiple of 60.

## Capabilities

```python
EnvironmentCapabilities(
    disable_internet=True,
    network_allowlist=True,
    network_allowlist_hostnames=True,
    # wildcards / IPs / CIDRs: False
    dynamic_network_policy=(no custom gateway),
    docker_compose=True,
)
```

Resource capabilities: CPU request + memory request → mapped to sandbox
`vcpus` / `memory_mb` (and storage → `disk_gb`).

## Start modes

`start(force_build)` always:

1. Deletes a previous sandbox if present + cleans gateway state.
2. `_setup_gateway()` → profile name or `None`.
3. Branches:

| Mode | Trigger | Sandbox image | Init | Follow-up |
|------|---------|---------------|------|-----------|
| Compose | `docker-compose.yaml` or extra compose | `islo-runner:latest` | `custom` + `docker` | `_start_compose()` |
| Prebuilt | `docker_image` + no force build | that image | `minimal` | mkdir / upload env |
| Dockerfile | `environment/Dockerfile` | `islo-runner:latest` | `custom` + `docker` | `_build_and_run_docker()` → `task-env` |
| Bare | fallback | `islo-runner:latest` | `minimal` | mkdir paths |

Constants:

- `_DEFAULT_IMAGE = "docker.io/library/islo-runner:latest"`
- `_DOCKER_CONTAINER_NAME = "task-env"`
- Poll: every 2s, up to 60 attempts, terminal statuses `failed|error|stopped|deleted`

## Gateway translation

`_gateway_config_from_network_policy`:

| `NetworkMode` | GatewayConfig |
|---------------|---------------|
| `PUBLIC` | allow + internet |
| `NO_NETWORK` | deny + `internet_enabled=False` |
| `ALLOWLIST` | deny + internet + allow rule per host |

Ephemeral profile name: `harbor-` + sanitized `session_id` (max 255).

Phase switches call `_apply_network_policy` → `_apply_gateway_config`, which
updates the profile, replaces rules, and sleeps `_GATEWAY_POLICY_PROPAGATION_DELAY_SEC` (2s).

Named profiles (`gateway_profile`) are never mutated or deleted by Harbor;
`dynamic_network_policy` is therefore `False` when a named/inline custom gateway is in use.

## Exec routing

```
exec(command)
  → _merge_env + _resolve_user
  → compose?  _compose_main_exec
  → docker container?  _docker_exec (docker exec task-env)
  → else  _sandbox_exec (SDK exec_and_wait on the VM)
```

## File transfer routing

For nested Docker/compose targets that are **not** on a bind mount / `/tests` /
`/solution`:

1. SDK upload/download to `/tmp/harbor_<uuid>` on the microVM
2. `docker cp` or `docker compose cp` into/out of the task container

Mount targets and Harbor log dirs use the SDK path directly.

## Compose internals (bundled only)

- Stage Harbor templates (`COMPOSE_BUILD_PATH` / `COMPOSE_PREBUILT_PATH`) plus
  resources + mounts overlays into `/harbor/compose`.
- Upload task `environment/` → `/harbor/environment`.
- Self-bind mounts rewrite host paths to equal container targets so VM paths match.
- Project name: sanitized session id.
- `docker compose -p <name> … build && up -d`, wait for `main`.

## Stop / attach

- `stop`: compose down or `docker stop task-env` → delete sandbox → cleanup ephemeral gateway.
- `attach`: `os.execvp("islo", ["islo", "use", sandbox, …])` into the right shell.

## Plugin (`harbor-islo`) deltas

Compared to bundled `islo.py`, the plugin:

- Has no `EnvironmentType` registration (use `import_path`)
- Lacks portable NetworkPolicy → gateway translation and dynamic phase updates
- Lacks `delete_after_seconds` and `ISLO_COMPUTE_URL`
- Uses legacy `init_capabilities` (including `core-gateway-proxy`) and a CA compose overlay
- Omits `ComposeServiceOpsMixin` / DinD sidecar helpers

Use the plugin when Islo needs to ship environment changes outside Harbor’s
release cadence; learn the full surface from the bundled file.

## Tests & examples

| Path | Purpose |
|------|---------|
| `tests/unit/environments/test_islo.py` | Gateway + policy unit coverage |
| `examples/configs/environments/islo/network-policy-demo.yaml` | Job config for Islo |
| `examples/tasks/islo-network-policy-demo/` | Phased allowlist demo task |
| `harbor-env/tests/integration/fixtures/*` | Compose / Dockerfile / prebuilt live fixtures |
