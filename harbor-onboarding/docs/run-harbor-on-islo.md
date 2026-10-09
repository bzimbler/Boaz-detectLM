# Run Harbor in an Islo environment

Step-by-step for pointing Harbor trials at Islo microVMs.

## 1. Prerequisites

- Python 3.12+ recommended (plugin requires 3.12+)
- An Islo API key (`ISLO_API_KEY` — Descope access key or session JWT)
- Agent provider keys as needed (e.g. `ANTHROPIC_API_KEY`)
- Optional: `islo` CLI on `PATH` if you want `attach()` / interactive `islo use`

## 2. Install path A — bundled (recommended)

```bash
pip install 'harbor[islo]'
# or from a harbor / harbor-fork checkout:
uv sync --extra islo

export ISLO_API_KEY=...
export ISLO_API_URL=https://api.islo.dev   # optional
export ISLO_COMPUTE_URL=...                # optional
```

Run:

```bash
harbor run \
  --env islo \
  --dataset terminal-bench@2.0 \
  --agent claude-code \
  --model anthropic/claude-opus-4-1 \
  --n-concurrent 4
```

Pass constructor kwargs:

```bash
harbor run --env islo \
  --environment-kwarg delete_after_seconds=7200 \
  --environment-kwarg gateway_profile=prod-apis \
  -c my-job.yaml
```

## 3. Install path B — plugin

```bash
pip install harbor-islo

harbor run \
  --env harbor_islo:IsloEnvironment \
  --dataset terminal-bench@2.0 \
  --agent claude-code \
  --model anthropic/claude-opus-4-1
```

YAML:

```yaml
environment:
  import_path: "harbor_islo:IsloEnvironment"
  kwargs:
    gateway:
      default_action: "deny"
      rules:
        - host_pattern: "api.openai.com"
          action: "allow"
          provider_key: "openai"
```

## 4. Network-policy demo (bundled)

From a Harbor or harbor-fork checkout:

```bash
export ISLO_API_KEY=...
export ANTHROPIC_API_KEY=...

uv run harbor run \
  -c examples/configs/environments/islo/network-policy-demo.yaml
```

What it shows:

- Setup / verifier phases stay on the public network baseline
- Agent phase switches to an allowlist (Anthropic + GitHub Gist hosts)
- Harbor creates an ephemeral gateway profile and updates it between phases

## 5. Gateway configuration rules

| Mode | When to use |
|------|-------------|
| Omit both | Harbor translates `NetworkPolicy` automatically (supports dynamic phases) |
| `gateway_profile` | Shared named profile already created in Islo control plane |
| `gateway` inline | Trial-local ephemeral profile with explicit rules |

**Never set both `gateway_profile` and `gateway`.**

**Never combine** Harbor `network_mode=allowlist|no-network` with a custom
gateway — the constructor raises so Harbor doesn’t claim a policy it can’t verify.

Islo allowlists accept **exact hostnames only** (no `*`, IPs, or CIDRs).

## 6. Lifecycle defaults

- Sandbox image default: `docker.io/library/islo-runner:latest`
- Auto-delete after **3600 seconds** (`delete_after_seconds`; `0` disables; otherwise multiple of 60)
- Create is cancel-shielded so a cancelled trial still cleans up the microVM
- Ephemeral gateway profiles are deleted on `stop`; named profiles are left alone

## 7. Mental model

```
harbor run --env islo
  → EnvironmentFactory → IsloEnvironment
  → gateway profile (named or ephemeral)
  → AsyncIslo.sandboxes.create_sandbox(...)
  → optional nested Docker / Compose inside the microVM
  → agent + verifier over SDK exec / two-hop file IO
  → delete sandbox + ephemeral gateway
```

Same Harbor job graphs as Docker — only the sandbox transport changes.
