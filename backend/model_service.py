"""Hooked distilgpt2 exposing attention as an editable variable.

Each transformer block's attention forward is replaced with an equivalent
eager implementation that (a) applies user-specified interventions to the
post-softmax attention matrix before the value mixdown and (b) captures the
resulting matrix for the UI. Only the block's own weight tensors (c_attn,
c_proj) are used, so this is robust across transformers versions.
"""
import math
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilgpt2"
MAX_INPUT_TOKENS = 48
TOP_K = 10


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
        self._lock = threading.Lock()
        self._interventions: list[dict] = []
        self._captured: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.model.transformer.h):
            block.attn.forward = self._make_forward(block.attn, i)

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
            w = self._apply(w, layer)
            self._captured[layer] = w.detach()
            out = (w @ v).transpose(1, 2).reshape(B, T, C)
            return attn.resid_dropout(attn.c_proj(out)), None

        return forward

    def _apply(self, att: torch.Tensor, layer: int) -> torch.Tensor:
        T = att.size(-1)
        for iv in self._interventions:
            h = int(iv.get("head", -1))
            if int(iv.get("layer", -1)) != layer or not (0 <= h < self.n_heads):
                continue
            op = iv.get("op")
            if op == "zero":
                att[:, h] = 0.0
            elif op == "scale":
                att[:, h] = att[:, h] * float(iv.get("value", 1.0))
            elif op in ("uniform", "self", "prev", "first"):
                att[:, h] = _pattern(op, T)
            elif op == "edit":
                self._apply_edit(att, h, T, iv)
        return att

    def _apply_edit(self, att, h, T, iv):
        # Clamp the given cells, then optionally rescale each touched row's
        # remaining cells so the row still sums to 1.
        fixed: dict[int, set[int]] = {}
        for c in iv.get("cells") or []:
            q, k = int(c["q"]), int(c["k"])
            if not (0 <= k <= q < T):
                continue
            att[:, h, q, k] = min(max(float(c["value"]), 0.0), 1.0)
            fixed.setdefault(q, set()).add(k)
        if not iv.get("renormalize", True):
            return
        for q, ks in fixed.items():
            row = att[0, h, q]
            fixed_sum = float(sum(row[k] for k in ks))
            others = [k for k in range(q + 1) if k not in ks]
            other_sum = float(sum(row[k] for k in others))
            target = max(0.0, 1.0 - fixed_sum)
            if other_sum > 1e-9:
                scale = target / other_sum
                for k in others:
                    att[:, h, q, k] *= scale

    def _run(self, ids: torch.Tensor, interventions: list[dict]):
        self._interventions = interventions
        self._captured = {}
        with torch.no_grad():
            logits = self.model(ids, use_cache=False).logits[0, -1]
        return logits, self._captured

    def analyze(self, text: str, interventions: list[dict] | None = None) -> dict:
        with self._lock:
            ids = self.tokenizer(text or "Hello", return_tensors="pt").input_ids
            ids = ids[:, -MAX_INPUT_TOKENS:]
            tokens = [self.tokenizer.decode([t]) for t in ids[0].tolist()]
            base_logits, base_att = self._run(ids, [])
            if interventions:
                int_logits, int_att = self._run(ids, interventions)
            else:
                int_logits, int_att = base_logits, base_att
            base_p = torch.softmax(base_logits, -1)
            int_p = torch.softmax(int_logits, -1)
            kl = torch.sum(
                base_p * (torch.log(base_p + 1e-12) - torch.log(int_p + 1e-12))
            ).item()
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
            attention = [
                [
                    [[round(x, 3) for x in row] for row in int_att[l][0, h].tolist()]
                    for h in range(self.n_heads)
                ]
                for l in range(self.n_layers)
            ]
            return {
                "tokens": tokens,
                "attention": attention,
                "dist": dist,
                "kl": round(kl, 4),
                "top_base": dist and max(dist, key=lambda d: d["base"])["token"],
                "top_int": dist and max(dist, key=lambda d: d["int"])["token"],
            }
