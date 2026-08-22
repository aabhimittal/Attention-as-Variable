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
scale or re-sharpen a head, blind it to a specific token, or **paint individual cells**
with your cursor — every edit re-runs the hooked forward pass and the next-token
distribution updates live, side by side with the un-edited baseline.

Beyond single edits it answers the question an attention viewer can't:
**which heads does this prediction actually depend on?**

| Feature | What it does |
|---|---|
| **Head sweep** | Ablates all 72 heads one at a time and ranks them by effect — a causal importance map, flip-flagged where the top token changes |
| **Contrast pairs** | Name two candidate tokens (` Mary` vs ` John`) and every edit is scored by the logit difference, the standard circuit-analysis metric |
| **Blind a head to a token** | Click a column label to knock out attention *to* that position — counterfactual masking without touching the prompt |
| **Temperature / top-k** | Re-sharpen or flatten a head's rows, or keep only its k strongest sources |
| **Intervened generation** | Continue the text with the edits live at every step — downstream effect, not just the next token |
| **Head fingerprints** | Per-head entropy, previous-token / self / sink scores and an inferred role label |
| **Permalinks** | The whole experiment (prompt, targets, every edit) round-trips through the URL |

## How it works

- **Backend** (`backend/`): FastAPI + PyTorch. Each GPT-2 block's attention forward is
  replaced with an equivalent eager implementation that applies interventions to the
  post-softmax attention matrix *before* the value mixdown, and captures the result.
  Only the block's own weights (`c_attn`, `c_proj`) are used, so it is robust across
  `transformers` versions.
- **Frontend** (`frontend/`): build-free vanilla JS. Canvas heatmaps as
  direct-manipulation UI; grouped bars compare baseline vs intervened distributions.

### API

- `POST /api/analyze` → `{text, interventions: [...], targets?: [str, str]}` where an
  intervention is `{layer, head, op, value?, positions?, cells?, renormalize?}` and
  `op ∈ zero | scale | uniform | self | prev | first | temp | topk | mask_key | only_key | edit`.
  Returns tokens, all post-intervention attention matrices, baseline + intervened
  top-token probabilities, KL(base‖intervened), per-head statistics and target metrics.
- `POST /api/sweep` → ablates every head individually; returns an L×H grid of KL /
  logit-difference deltas plus the ten most influential heads.
- `POST /api/generate` → greedy or sampled continuation, run clean and intervened.
- `GET /api/model_info`, `GET /api/health`

Every op is bounded and sanitized server-side: non-finite values, out-of-range
layers/heads, above-diagonal cells and oversized payloads are rejected or dropped
rather than reaching the forward pass, and edited rows are renormalized back to a
valid distribution (see `tests/test_edge_cases.py`).

## Run locally

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn backend.app:app --port 7860
# open http://localhost:7860
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

82 checks covering adversarial payloads (NaN/∞ values, malformed cells, unknown ops,
oversized requests), degenerate prompts (empty, control characters, emoji, RTL
overrides, single-token, over-length), numerical invariants (row normalization,
forward-pass fidelity to stock GPT-2 within 1e-4), cross-request state leakage,
concurrency, and the HTTP contract.

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
Try: `When Mary and John went to the store, John gave a drink to` with the contrast
pair ` Mary` / ` John`, then hit **Sweep all heads** — the heads at the top of the
ranking are the ones carrying the name, and knocking one out moves the logit
difference you're scoring. That is the name-mover end of the IOI circuit, found
causally rather than by eyeballing heatmaps. That loop — hypothesis, edit, shifted
distribution — is the whole point.

## License

MIT
