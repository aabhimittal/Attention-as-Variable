---
title: Attention as a Variable
emoji: 🎛️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Attention as a Variable

Attention viewers show you what a transformer did. This one lets you **change it**.

A real forward pass of `distilgpt2` is rendered as 6×12 editable attention matrices.
Zero out a head, force a pattern (uniform / self / previous-token / attention-sink),
scale a head, or **paint individual cells** with your cursor — every edit re-runs the
hooked forward pass and the next-token distribution updates live, side by side with
the un-edited baseline (plus the KL divergence between them).

## How it works

- **Backend** (`backend/`): FastAPI + PyTorch. Each GPT-2 block's attention forward is
  replaced with an equivalent eager implementation that applies interventions to the
  post-softmax attention matrix *before* the value mixdown, and captures the result.
  Only the block's own weights (`c_attn`, `c_proj`) are used, so it is robust across
  `transformers` versions.
- **Frontend** (`frontend/`): build-free vanilla JS. Canvas heatmaps as
  direct-manipulation UI; grouped bars compare baseline vs intervened distributions.

### API

- `POST /api/analyze` → `{text, interventions: [{layer, head, op, value?, cells?, renormalize?}]}`
  where `op ∈ zero | scale | uniform | self | prev | first | edit`.
  Returns tokens, all post-intervention attention matrices, baseline + intervened
  top-token probabilities, and KL(base‖intervened).
- `GET /api/model_info`, `GET /api/health`

## Run locally

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn backend.app:app --port 7860
# open http://localhost:7860
```

## Deploy

One codebase, three targets:

| Target | What runs | How |
|---|---|---|
| **Hugging Face Space** | full app (API + UI) | Docker Space: push this repo as-is (the YAML frontmatter above + `Dockerfile` are the config) |
| **Vercel** | static frontend | `vercel deploy` — `vercel.json` serves `frontend/`; the UI talks to the Space API |
| **GitHub Pages** | static frontend | `.github/workflows/pages.yml` deploys `frontend/` on push to `main` (enable Pages → Source: GitHub Actions) |

The static deployments auto-detect that no same-origin backend exists and fall back
to the hosted Space API (editable in the header, persisted in `localStorage`).

## Why interventions?

Looking at attention tells you what correlates; *editing* it tells you what matters.
Try: `When Mary and John went to the store, John gave a drink to` — find the heads
whose knockout flips the prediction away from ` Mary`, and you've located the
name-mover heads of the IOI circuit. That causal loop — hypothesis, edit, shifted
distribution — is the pedagogical point.

## License

MIT
