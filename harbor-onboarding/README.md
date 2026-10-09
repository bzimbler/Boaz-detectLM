# Harbor × Islo Onboarding

Interactive onboarding site for the [Harbor](https://github.com/harbor-framework/harbor) agent-evaluation framework, plus Islo’s fork and the option to run Harbor trials on Islo microVMs.

## Open the site

### iPhone (recommended)

Use the **single file** `Harbor-Islo-Onboarding.html` — styles and scripts are
inlined, so HTML Viewer / Files apps that only open one file still look correct.

1. Download that file into **Files** (AirDrop, Safari download, etc.).
2. Tap it → open with **HTML Viewer**, **Lookin**, or Safari if offered.
3. You should see teal branding, nav, and formatted sections — not plain text.

Avoid opening only `index.html` on iPhone: many apps ignore sibling `styles.css`
/ `app.js`, which is why it looked unstyled.

### Computer

```bash
# From this directory
python3 -m http.server 8765
# then open http://localhost:8765
```

Or open `Harbor-Islo-Onboarding.html` / `index.html` in a desktop browser.

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
