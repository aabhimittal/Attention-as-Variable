"""Hooked causal LM exposing attention as an editable, queryable variable.

Each transformer block's attention forward is replaced with an equivalent
eager implementation that (a) applies user-specified interventions to the
post-softmax attention matrix before the value mixdown and (b) captures the
resulting matrix for the UI. Only the block's own weight tensors (c_attn,
c_proj) are used, so this is robust across transformers versions.

Beyond single-head editing this module supports:
  * causal head-attribution sweeps  (ablate every head, rank by effect)
  * target-token metrics            (prob / logit / logit-difference)
  * intervened text generation      (downstream effect, not just next token)
  * per-head behavioural statistics (entropy, prev/self/sink scores, role)
"""
import math
import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("AAV_MODEL", "distilgpt2")
MAX_INPUT_TOKENS = int(os.getenv("AAV_MAX_TOKENS", "48"))
MAX_NEW_TOKENS = 32
TOP_K = 10

# Everything the frontend may ask for on a single head.
PATTERN_OPS = ("uniform", "self", "prev", "first")
OPS = PATTERN_OPS + ("zero", "scale", "edit", "temp", "topk", "mask_key", "only_key")

# Hard bounds: an intervention must never be able to produce a non-finite
# forward pass, and must never be able to make the payload unbounded.
MAX_INTERVENTIONS = 64
MAX_CELLS = 4096
ATT_CLAMP = 1e4


def _finite(x, default: float = 0.0) -> float:
    """Coerce anything the wire may carry into a finite float."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _pattern(op: str, T: int) -> torch.Tensor:
    if op == "uniform":
        m = torch.tril(torch.ones(T, T))
        return m / m.sum(-1, keepdim=True)
    if op == "self":
        return torch.eye(T)
    if op == "prev":
        p = torch.zeros(T, T)
        p[0, 0] = 1.0
        if T > 1:
            p[torch.arange(1, T), torch.arange(0, T - 1)] = 1.0
        return p
    if op == "first":
        p = torch.zeros(T, T)
        p[:, 0] = 1.0
        return p
    raise ValueError(f"unknown pattern op: {op}")


def _renorm_rows(rows: torch.Tensor, T: int) -> torch.Tensor:
    """Rescale each causal row to sum to 1; rows that went empty fall back
    to the position's own token so the pass stays a valid distribution."""
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=rows.device))
    rows = rows * causal
    s = rows.sum(-1, keepdim=True)
    dead = s.squeeze(-1) <= 1e-9
    if dead.any():
        eye = torch.eye(T, device=rows.device).expand_as(rows)
        rows = torch.where(dead.unsqueeze(-1), eye, rows)
        s = rows.sum(-1, keepdim=True)
    return rows / s.clamp(min=1e-9)


class InterventionModel:
    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager"
        )
        self.model.eval()
        self.model_id = model_id
        self.n_layers = self.model.config.n_layer
        self.n_heads = self.model.config.n_head
        self._lock = threading.RLock()
        self._interventions: list[dict] = []
        self._captured: dict[int, torch.Tensor] = {}
        self._capture = True
        for i, block in enumerate(self.model.transformer.h):
            block.attn.forward = self._make_forward(block.attn, i)

    # ---------------- hooked forward ----------------

    def _make_forward(self, attn, layer: int):
        n_heads = getattr(attn, "num_heads", None) or attn.config.num_attention_heads
        head_dim = getattr(attn, "head_dim", None) or attn.config.hidden_size // n_heads

        def forward(hidden_states, *args, **kwargs):
            B, T, C = hidden_states.shape
            q, k, v = attn.c_attn(hidden_states).split(attn.split_size, dim=2)

            def heads(x):
                return x.view(B, T, n_heads, head_dim).transpose(1, 2)

            q, k, v = heads(q), heads(k), heads(v)
            w = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=w.device))
            w = w.masked_fill(~causal, torch.finfo(w.dtype).min)
            w = torch.softmax(w, dim=-1)
            if self._interventions:
                w = self._apply(w, layer)
                # An edited attention matrix is user input: never let a NaN,
                # an infinity or an absurd magnitude reach the value mixdown.
                w = torch.nan_to_num(w, nan=0.0, posinf=ATT_CLAMP, neginf=0.0)
                w = w.clamp(0.0, ATT_CLAMP)
            if self._capture:
                self._captured[layer] = w.detach()
            out = (w @ v).transpose(1, 2).reshape(B, T, C)
            return attn.resid_dropout(attn.c_proj(out)), None

        return forward

    # ---------------- interventions ----------------

    def _apply(self, att: torch.Tensor, layer: int) -> torch.Tensor:
        T = att.size(-1)
        for iv in self._interventions:
            try:
                h = int(iv.get("head", -1))
                lay = int(iv.get("layer", -1))
            except (TypeError, ValueError):
                continue
            if lay != layer or not (0 <= h < self.n_heads):
                continue
            op = iv.get("op")
            if op == "zero":
                att[:, h] = 0.0
            elif op == "scale":
                att[:, h] = att[:, h] * max(0.0, min(_finite(iv.get("value"), 1.0), 100.0))
            elif op in PATTERN_OPS:
                att[:, h] = _pattern(op, T).to(att.device)
            elif op == "temp":
                att[:, h] = self._temp(att[:, h], T, _finite(iv.get("value"), 1.0))
            elif op == "topk":
                att[:, h] = self._topk(att[:, h], T, iv.get("value"))
            elif op in ("mask_key", "only_key"):
                att[:, h] = self._key_gate(att[:, h], T, iv, keep=op == "only_key")
            elif op == "edit":
                self._apply_edit(att, h, T, iv)
        return att

    @staticmethod
    def _temp(rows: torch.Tensor, T: int, tau: float) -> torch.Tensor:
        """Re-sharpen (tau<1) or flatten (tau>1) each attention row."""
        tau = max(0.05, min(tau, 20.0))
        return _renorm_rows(rows.clamp(min=0).pow(1.0 / tau), T)

    @staticmethod
    def _topk(rows: torch.Tensor, T: int, value) -> torch.Tensor:
        """Keep only the k strongest sources per row, renormalized."""
        k = max(1, min(int(_finite(value, 1.0)), T))
        thresh = rows.topk(k, dim=-1).values[..., -1:]
        return _renorm_rows(torch.where(rows >= thresh, rows, torch.zeros_like(rows)), T)

    @staticmethod
    def _key_gate(rows: torch.Tensor, T: int, iv: dict, keep: bool) -> torch.Tensor:
        """Blind the head to specific source positions (mask_key), or to
        everything except them (only_key). Renormalized either way."""
        raw = iv.get("positions")
        if raw is None:
            raw = [iv.get("value", 0)]
        pos = {int(_finite(p, -1)) for p in raw if -1 < _finite(p, -1) < T}
        if not pos:
            return rows
        sel = torch.zeros(T, dtype=torch.bool, device=rows.device)
        sel[list(pos)] = True
        mask = sel if keep else ~sel
        return _renorm_rows(rows * mask, T)

    def _apply_edit(self, att, h, T, iv):
        """Clamp the given cells, then optionally rescale each touched row's
        remaining cells so the row still sums to 1."""
        cells = (iv.get("cells") or [])[:MAX_CELLS]
        fixed: dict[int, set[int]] = {}
        for c in cells:
            try:
                q, k = int(c["q"]), int(c["k"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (0 <= k <= q < T):
                continue
            att[:, h, q, k] = min(max(_finite(c.get("value"), 0.0), 0.0), 1.0)
            fixed.setdefault(q, set()).add(k)
        if not fixed or not iv.get("renormalize", True):
            return
        for q, ks in fixed.items():
            row = att[0, h, q]
            fixed_sum = float(row[list(ks)].sum())
            others = [k for k in range(q + 1) if k not in ks]
            if not others:
                continue
            other_sum = float(row[others].sum())
            target = max(0.0, 1.0 - fixed_sum)
            if other_sum > 1e-9:
                att[:, h, q, others] *= target / other_sum
            elif target > 0:
                att[:, h, q, others] = target / len(others)

    # ---------------- forward helpers ----------------

    def _run(self, ids: torch.Tensor, interventions: list[dict], capture: bool = True):
        self._interventions = interventions or []
        self._captured = {}
        self._capture = capture
        try:
            with torch.no_grad():
                logits = self.model(ids, use_cache=False).logits[0, -1]
        finally:
            self._interventions = []
            self._capture = True
        return torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4), self._captured

    def encode(self, text: str) -> torch.Tensor:
        """Tokenize defensively: empty / whitespace / control-only input still
        has to produce at least one valid position."""
        ids = self.tokenizer(text or "", return_tensors="pt").input_ids
        if ids.numel() == 0:
            fallback = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
            ids = torch.tensor([[fallback]], dtype=torch.long)
        return ids[:, -MAX_INPUT_TOKENS:]

    def _target_ids(self, targets: list[str] | None) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for t in (targets or [])[:8]:
            if not isinstance(t, str) or not t:
                continue
            enc = self.tokenizer(t).input_ids
            if enc:
                out.append((t, int(enc[0])))
        return out

    @staticmethod
    def _kl(base_p: torch.Tensor, int_p: torch.Tensor) -> float:
        kl = torch.sum(
            base_p * (torch.log(base_p.clamp(min=1e-12)) - torch.log(int_p.clamp(min=1e-12)))
        ).item()
        return round(kl if math.isfinite(kl) else 0.0, 4)

    def _targets_block(self, pairs, base_logits, int_logits, base_p, int_p) -> dict:
        rows = [
            {
                "token": tok,
                "id": tid,
                "base": round(base_p[tid].item(), 6),
                "int": round(int_p[tid].item(), 6),
                "base_logit": round(base_logits[tid].item(), 4),
                "int_logit": round(int_logits[tid].item(), 4),
            }
            for tok, tid in pairs
        ]
        out = {"targets": rows}
        if len(pairs) >= 2:
            a, b = pairs[0][1], pairs[1][1]
            bd = (base_logits[a] - base_logits[b]).item()
            idf = (int_logits[a] - int_logits[b]).item()
            out["logit_diff"] = {
                "base": round(bd, 4),
                "int": round(idf, 4),
                "delta": round(idf - bd, 4),
            }
        return out

    # ---------------- head statistics ----------------

    def _head_stats(self, att: dict[int, torch.Tensor]) -> list[list[dict]]:
        stats = []
        for l in range(self.n_layers):
            m = att.get(l)
            row = []
            for h in range(self.n_heads):
                row.append(self._one_head_stats(m[0, h]) if m is not None else {})
            stats.append(row)
        return stats

    @staticmethod
    def _one_head_stats(a: torch.Tensor) -> dict:
        T = a.size(0)
        idx = torch.arange(T)
        p = a.clamp(min=1e-12)
        ent = -(a * p.log()).sum(-1)
        norm = torch.log(torch.arange(1, T + 1).float()).clamp(min=1e-9)
        ent_n = float((ent[1:] / norm[1:]).mean()) if T > 1 else 0.0
        prev = float(a[idx[1:], idx[1:] - 1].mean()) if T > 1 else 0.0
        first = float(a[1:, 0].mean()) if T > 1 else 1.0
        slf = float(a[idx[1:], idx[1:]].mean()) if T > 1 else 1.0
        role = "mixed"
        if prev > 0.5:
            role = "previous-token"
        elif slf > 0.5:
            role = "self / identity"
        elif first > 0.5:
            role = "attention sink"
        elif ent_n > 0.85:
            role = "diffuse"
        return {
            "entropy": round(max(0.0, min(ent_n, 1.0)), 3),
            "prev": round(prev, 3),
            "first": round(first, 3),
            "self": round(slf, 3),
            "role": role,
        }

    # ---------------- public API ----------------

    def analyze(
        self,
        text: str,
        interventions: list[dict] | None = None,
        targets: list[str] | None = None,
        include_stats: bool = True,
    ) -> dict:
        interventions = (interventions or [])[:MAX_INTERVENTIONS]
        with self._lock:
            ids = self.encode(text)
            tokens = [self.tokenizer.decode([t]) for t in ids[0].tolist()]
            base_logits, base_att = self._run(ids, [])
            if interventions:
                int_logits, int_att = self._run(ids, interventions)
            else:
                int_logits, int_att = base_logits, base_att
            base_p = torch.softmax(base_logits, -1)
            int_p = torch.softmax(int_logits, -1)
            union = list(
                set(base_p.topk(TOP_K).indices.tolist())
                | set(int_p.topk(TOP_K).indices.tolist())
            )
            union.sort(key=lambda i: -max(base_p[i].item(), int_p[i].item()))
            dist = [
                {
                    "token": self.tokenizer.decode([i]),
                    "base": round(base_p[i].item(), 5),
                    "int": round(int_p[i].item(), 5),
                }
                for i in union
            ]
            out = {
                "tokens": tokens,
                "attention": self._serialize(int_att),
                "dist": dist,
                "kl": self._kl(base_p, int_p),
                "top_base": dist and max(dist, key=lambda d: d["base"])["token"],
                "top_int": dist and max(dist, key=lambda d: d["int"])["token"],
            }
            if include_stats:
                out["head_stats"] = self._head_stats(base_att)
            pairs = self._target_ids(targets)
            if pairs:
                out.update(self._targets_block(pairs, base_logits, int_logits, base_p, int_p))
            return out

    def _serialize(self, att: dict[int, torch.Tensor]) -> list:
        return [
            [
                [[round(x, 3) for x in row] for row in att[l][0, h].tolist()]
                for h in range(self.n_heads)
            ]
            for l in range(self.n_layers)
        ]

    def sweep(
        self,
        text: str,
        op: str = "zero",
        value: float | None = None,
        targets: list[str] | None = None,
        base_interventions: list[dict] | None = None,
    ) -> dict:
        """Causal head-attribution: ablate each head on its own and rank the
        heads by how much the next-token distribution moves."""
        if op not in ("zero", "uniform", "self", "prev", "first", "scale", "temp"):
            raise ValueError(f"unsupported sweep op: {op}")
        base_interventions = (base_interventions or [])[:MAX_INTERVENTIONS]
        with self._lock:
            ids = self.encode(text)
            base_logits, base_att = self._run(ids, base_interventions, capture=True)
            base_p = torch.softmax(base_logits, -1)
            pairs = self._target_ids(targets)
            base_diff = (
                (base_logits[pairs[0][1]] - base_logits[pairs[1][1]]).item()
                if len(pairs) >= 2
                else None
            )
            top_base = int(base_p.argmax())
            grid, flat = [], []
            for l in range(self.n_layers):
                row = []
                for h in range(self.n_heads):
                    ivs = base_interventions + [
                        {"layer": l, "head": h, "op": op, "value": value}
                    ]
                    lg, _ = self._run(ids, ivs, capture=False)
                    p = torch.softmax(lg, -1)
                    cell = {
                        "kl": self._kl(base_p, p),
                        "flip": int(p.argmax()) != top_base,
                    }
                    if base_diff is not None:
                        cell["logit_diff_delta"] = round(
                            (lg[pairs[0][1]] - lg[pairs[1][1]]).item() - base_diff, 4
                        )
                    if pairs:
                        cell["target_delta"] = round(
                            (p[pairs[0][1]] - base_p[pairs[0][1]]).item(), 5
                        )
                    row.append(cell)
                    flat.append({"layer": l, "head": h, **cell})
                grid.append(row)
            key = "logit_diff_delta" if base_diff is not None else "kl"
            flat.sort(key=lambda c: -abs(c.get(key, 0.0)))
            return {
                "op": op,
                "tokens": [self.tokenizer.decode([t]) for t in ids[0].tolist()],
                "grid": grid,
                "top": flat[:10],
                "metric": key,
                "base_logit_diff": None if base_diff is None else round(base_diff, 4),
                "head_stats": self._head_stats(base_att),
            }

    def generate(
        self,
        text: str,
        interventions: list[dict] | None = None,
        max_new_tokens: int = 12,
        temperature: float = 0.0,
    ) -> dict:
        """Greedy (temperature 0) or sampled continuation, run twice: once
        clean and once with the interventions live at every step."""
        n = max(1, min(int(_finite(max_new_tokens, 12)), MAX_NEW_TOKENS))
        temp = max(0.0, min(_finite(temperature, 0.0), 2.0))
        interventions = (interventions or [])[:MAX_INTERVENTIONS]
        with self._lock:
            ids = self.encode(text)
            prompt = self.tokenizer.decode(ids[0].tolist())
            return {
                "prompt": prompt,
                "baseline": self._continue(ids, [], n, temp),
                "intervened": self._continue(ids, interventions, n, temp),
                "max_new_tokens": n,
            }

    def _continue(self, ids, interventions, n, temp) -> str:
        cur = ids
        gen: list[int] = []
        eos = self.tokenizer.eos_token_id
        for _ in range(n):
            logits, _ = self._run(cur[:, -MAX_INPUT_TOKENS:], interventions, capture=False)
            if temp <= 0:
                nxt = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temp, -1)
                nxt = int(torch.multinomial(probs, 1))
            if nxt == eos:
                break
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=cur.dtype)], dim=1)
        return self.tokenizer.decode(gen)
