# Harbor × Islo Onboarding

Interactive onboarding site for the [Harbor](https://github.com/harbor-framework/harbor) agent-evaluation framework, plus Islo’s fork and the option to run Harbor trials on Islo microVMs.

## Open the site

```bash
# From this directory
python3 -m http.server 8765
# then open http://localhost:8765
```

Or open `index.html` directly in a browser.

## What’s inside

| Path | Purpose |
|------|---------|
| `index.html` | Interactive onboarding site (concepts, run paths, Islo code) |
| `styles.css` | Visual system |
| `app.js` | Nav, section reveal, run-path toggles, code explorer |
| `docs/` | Markdown deep-dives (same spirit as islo-web-api `docs/`) |

## Source repos

- Upstream Harbor: https://github.com/harbor-framework/harbor
- Islo fork: https://github.com/islo-labs/harbor-fork
- Standalone Islo plugin: https://github.com/islo-labs/harbor-env
- Docs: https://harborframework.com/docs
