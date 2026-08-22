"""Industrial edge-case suite for the attention-intervention service.

Covers the failure classes that actually bite a hosted interpretability
service: adversarial payloads, degenerate prompts, numerical blow-ups,
cross-request state leakage and concurrency.
"""
import json
import math
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.model_service import MAX_INPUT_TOKENS, InterventionModel

PROMPT = "When Mary and John went to the store, John gave a drink to"


@pytest.fixture(scope="session")
def model():
    return InterventionModel()


@pytest.fixture(scope="session")
def client(model):
    app_module._model = model  # reuse the loaded weights; no second download
    return TestClient(app_module.app)


def rows_of(att, l, h):
    return att[l][h]


def causal_row_sums(att, l, h):
    """Sum of each causal row (row q covers keys 0..q)."""
    return [sum(row[: q + 1]) for q, row in enumerate(rows_of(att, l, h))]


# ---------------- fidelity ----------------


def test_hooked_forward_matches_stock_model(model):
    from transformers import AutoModelForCausalLM

    stock = AutoModelForCausalLM.from_pretrained(
        model.model_id, attn_implementation="eager"
    ).eval()
    ids = model.encode(PROMPT)
    with torch.no_grad():
        ref = stock(ids, use_cache=False).logits[0, -1]
    got, _ = model._run(ids, [])
    assert torch.allclose(ref, got, atol=1e-4)


def test_no_intervention_is_a_no_op(model):
    out = model.analyze(PROMPT)
    assert out["kl"] == 0.0
    assert out["top_base"] == out["top_int"]


# ---------------- degenerate prompts ----------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\n\t  ",
        "\x00\x01\x02",
        "a",
        "🙂",
        "東京",
        "🇯🇵👨‍👩‍👧‍👦",
        "‮gnitset",  # RTL override
        "'; DROP TABLE heads;--",
        "<script>alert(1)</script>",
        "x" * 3000,
    ],
)
def test_degenerate_prompts_stay_serializable(model, text):
    out = model.analyze(text, [{"layer": 0, "head": 0, "op": "zero"}])
    assert 1 <= len(out["tokens"]) <= MAX_INPUT_TOKENS
    assert math.isfinite(out["kl"]) and out["kl"] >= 0
    json.dumps(out)  # must survive lone surrogates / partial byte tokens


def test_long_prompt_is_truncated(model):
    out = model.analyze(" ".join(["token"] * 400))
    assert len(out["tokens"]) == MAX_INPUT_TOKENS
    assert len(out["attention"][0][0]) == MAX_INPUT_TOKENS


def test_single_token_prompt_survives_every_op(model):
    for op in ("zero", "uniform", "self", "prev", "first", "temp", "topk", "mask_key",
               "only_key", "scale"):
        out = model.analyze("a", [{"layer": 0, "head": 0, "op": op, "value": 0.5}])
        assert math.isfinite(out["kl"]), op
        assert len(out["tokens"]) == 1


# ---------------- adversarial numerics ----------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -5.0, 1e30])
def test_pathological_scale_never_produces_nan(model, value):
    out = model.analyze(PROMPT, [{"layer": 3, "head": 2, "op": "scale", "value": value}])
    assert math.isfinite(out["kl"])
    assert all(math.isfinite(d["int"]) for d in out["dist"])
    assert abs(sum(d["int"] for d in out["dist"])) <= 1.0001


@pytest.mark.parametrize("value", [0.0, float("nan"), 1e9, -1.0])
def test_pathological_temperature_keeps_rows_normalized(model, value):
    out = model.analyze(PROMPT, [{"layer": 1, "head": 1, "op": "temp", "value": value}])
    for s in causal_row_sums(out["attention"], 1, 1):
        assert abs(s - 1.0) < 0.02
    assert math.isfinite(out["kl"])


def test_nan_cell_value_is_ignored_not_propagated(model):
    out = model.analyze(
        PROMPT,
        [{
            "layer": 2, "head": 4, "op": "edit",
            "cells": [{"q": 3, "k": 1, "value": float("nan")}],
        }],
    )
    assert math.isfinite(out["kl"])
    assert all(math.isfinite(v) for row in rows_of(out["attention"], 2, 4) for v in row)


# ---------------- malformed intervention payloads ----------------


def test_out_of_range_layer_and_head_are_ignored(model):
    ivs = [
        {"layer": 99, "head": 0, "op": "zero"},
        {"layer": 0, "head": 99, "op": "zero"},
        {"layer": -1, "head": -1, "op": "zero"},
    ]
    assert model.analyze(PROMPT, ivs)["kl"] == 0.0


def test_non_numeric_layer_is_ignored(model):
    out = model.analyze(PROMPT, [{"layer": "one", "head": None, "op": "zero"}])
    assert out["kl"] == 0.0


def test_upper_triangle_and_oob_cells_are_dropped(model):
    out = model.analyze(
        PROMPT,
        [{
            "layer": 1, "head": 0, "op": "edit",
            "cells": [
                {"q": 2, "k": 5, "value": 1.0},     # above the causal diagonal
                {"q": 999, "k": 0, "value": 1.0},   # past the sequence
                {"q": -1, "k": 0, "value": 1.0},
                {"q": 4, "k": "x", "value": 1.0},   # malformed
            ],
        }],
    )
    assert out["kl"] == 0.0
    assert rows_of(out["attention"], 1, 0)[2][5] == 0.0


def test_empty_and_missing_cell_lists(model):
    for cells in ([], None):
        out = model.analyze(PROMPT, [{"layer": 0, "head": 0, "op": "edit", "cells": cells}])
        assert out["kl"] == 0.0


def test_intervention_list_is_capped(model):
    many = [{"layer": 0, "head": 0, "op": "zero"}] * 500
    assert math.isfinite(model.analyze(PROMPT, many)["kl"])


def test_conflicting_ops_on_one_head_are_last_write_wins(model):
    ivs = [
        {"layer": 2, "head": 3, "op": "uniform"},
        {"layer": 2, "head": 3, "op": "first"},
    ]
    att = model.analyze(PROMPT, ivs)["attention"]
    row = rows_of(att, 2, 3)[4]
    assert row[0] == 1.0 and sum(row[1:5]) == 0.0


# ---------------- intervention semantics ----------------


def test_edit_renormalizes_rows_to_one(model):
    out = model.analyze(
        PROMPT,
        [{
            "layer": 4, "head": 5, "op": "edit",
            "cells": [{"q": 6, "k": 2, "value": 0.9}], "renormalize": True,
        }],
    )
    assert abs(causal_row_sums(out["attention"], 4, 5)[6] - 1.0) < 0.02


def test_edit_without_renormalize_leaves_row_unnormalized(model):
    out = model.analyze(
        PROMPT,
        [{
            "layer": 4, "head": 5, "op": "edit",
            "cells": [{"q": 6, "k": 2, "value": 1.0}], "renormalize": False,
        }],
    )
    assert causal_row_sums(out["attention"], 4, 5)[6] > 1.2


def test_edit_saturating_a_row_leaves_a_valid_distribution(model):
    """Clamping every cell of a row to 0 must not produce a 0/0 row."""
    cells = [{"q": 5, "k": k, "value": 0.0} for k in range(6)]
    out = model.analyze(
        PROMPT, [{"layer": 0, "head": 0, "op": "edit", "cells": cells}]
    )
    assert math.isfinite(out["kl"])


@pytest.mark.parametrize("op", ["uniform", "self", "prev", "first"])
def test_patterns_are_exact_and_normalized(model, op):
    att = model.analyze(PROMPT, [{"layer": 0, "head": 7, "op": op}])["attention"]
    for s in causal_row_sums(att, 0, 7):
        assert abs(s - 1.0) < 0.01
    rows = rows_of(att, 0, 7)
    if op == "prev":
        assert rows[3][2] == 1.0
    if op == "self":
        assert rows[3][3] == 1.0
    if op == "first":
        assert rows[3][0] == 1.0


def test_topk_keeps_exactly_k_sources(model):
    att = model.analyze(PROMPT, [{"layer": 3, "head": 3, "op": "topk", "value": 2}])["attention"]
    row = rows_of(att, 3, 3)[8]
    assert sum(1 for v in row[:9] if v > 0) <= 2
    assert abs(sum(row[:9]) - 1.0) < 0.02


def test_topk_larger_than_sequence_is_a_noop(model):
    out = model.analyze(PROMPT, [{"layer": 3, "head": 3, "op": "topk", "value": 10_000}])
    assert out["kl"] < 1e-6


def test_mask_key_blinds_the_head_to_a_position(model):
    att = model.analyze(
        PROMPT, [{"layer": 2, "head": 2, "op": "mask_key", "positions": [1, 3]}]
    )["attention"]
    for row in rows_of(att, 2, 2)[4:]:
        assert row[1] == 0.0 and row[3] == 0.0
    assert abs(causal_row_sums(att, 2, 2)[6] - 1.0) < 0.02


def test_only_key_restricts_to_the_named_positions(model):
    att = model.analyze(
        PROMPT, [{"layer": 2, "head": 2, "op": "only_key", "positions": [0]}]
    )["attention"]
    assert rows_of(att, 2, 2)[5][0] == 1.0


def test_key_gate_with_no_valid_positions_is_a_noop(model):
    out = model.analyze(
        PROMPT, [{"layer": 2, "head": 2, "op": "mask_key", "positions": [999, -4]}]
    )
    assert out["kl"] == 0.0


def test_only_key_on_an_unreachable_position_falls_back_safely(model):
    """Row 0 can only see position 0; only_key([4]) leaves it empty."""
    out = model.analyze(
        PROMPT, [{"layer": 0, "head": 0, "op": "only_key", "positions": [4]}]
    )
    assert math.isfinite(out["kl"])
    assert abs(causal_row_sums(out["attention"], 0, 0)[0] - 1.0) < 0.02


def test_zeroing_a_head_actually_moves_the_prediction(model):
    kls = [
        model.analyze(PROMPT, [{"layer": l, "head": h, "op": "zero"}])["kl"]
        for l in (4, 5) for h in range(4)
    ]
    assert max(kls) > 0.0


# ---------------- state hygiene ----------------


def test_interventions_do_not_leak_into_later_requests(model):
    clean = model.analyze(PROMPT)["dist"][0]["base"]
    model.analyze(PROMPT, [{"layer": 0, "head": 0, "op": "zero"}])
    assert model.analyze(PROMPT)["dist"][0]["base"] == clean
    assert model.analyze(PROMPT)["kl"] == 0.0


def test_repeated_identical_requests_are_deterministic(model):
    ivs = [{"layer": 1, "head": 2, "op": "temp", "value": 0.4}]
    a = model.analyze(PROMPT, ivs)
    b = model.analyze(PROMPT, ivs)
    assert a["dist"] == b["dist"] and a["kl"] == b["kl"]


def test_concurrent_requests_do_not_cross_contaminate(model):
    expected = {
        h: model.analyze(PROMPT, [{"layer": 5, "head": h, "op": "zero"}])["kl"]
        for h in range(4)
    }
    seen, errors = {}, []
    lock = threading.Lock()

    def work(h):
        try:
            kl = model.analyze(PROMPT, [{"layer": 5, "head": h, "op": "zero"}])["kl"]
            with lock:
                seen[h] = kl
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(h,)) for _ in range(2) for h in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors and seen == expected


# ---------------- head statistics ----------------


def test_head_stats_shape_and_ranges(model):
    stats = model.analyze(PROMPT)["head_stats"]
    assert len(stats) == model.n_layers and len(stats[0]) == model.n_heads
    flat = [s for row in stats for s in row]
    assert all(0.0 <= s["entropy"] <= 1.0 for s in flat)
    assert all(0.0 <= s["prev"] <= 1.0 for s in flat)
    assert any(s["role"] != "mixed" for s in flat)


def test_head_stats_detect_a_forced_previous_token_head(model):
    out = model.analyze(PROMPT, [{"layer": 0, "head": 0, "op": "prev"}], include_stats=True)
    forced = model._one_head_stats(torch.tensor(rows_of(out["attention"], 0, 0)))
    assert forced["role"] == "previous-token" and forced["prev"] > 0.99


# ---------------- targets ----------------


def test_target_metrics_and_logit_difference(model):
    out = model.analyze(PROMPT, targets=[" Mary", " John"])
    assert len(out["targets"]) == 2
    ld = out["logit_diff"]
    assert ld["delta"] == 0.0  # no interventions yet
    assert math.isfinite(ld["base"])


def test_intervention_moves_the_logit_difference(model):
    ivs = [{"layer": l, "head": h, "op": "zero"} for l in range(model.n_layers)
           for h in range(model.n_heads)]
    out = model.analyze(PROMPT, ivs, targets=[" Mary", " John"])
    assert out["logit_diff"]["delta"] != 0.0


@pytest.mark.parametrize("targets", [[], [""], ["", None, 5], ["only-one"]])
def test_degenerate_targets_do_not_break_the_response(model, targets):
    out = model.analyze(PROMPT, targets=targets)
    assert "logit_diff" not in out or "base" in out["logit_diff"]


# ---------------- sweep ----------------


def test_sweep_grid_shape_and_ranking(model):
    out = model.sweep("The capital of France is", op="zero")
    assert len(out["grid"]) == model.n_layers
    assert len(out["grid"][0]) == model.n_heads
    assert all(c["kl"] >= 0 for row in out["grid"] for c in row)
    assert out["metric"] == "kl"
    kls = [c["kl"] for c in out["top"]]
    assert kls == sorted(kls, reverse=True) and kls[0] > 0


def test_sweep_with_targets_switches_metric(model):
    out = model.sweep(PROMPT, targets=[" Mary", " John"])
    assert out["metric"] == "logit_diff_delta"
    assert math.isfinite(out["base_logit_diff"])
    assert all("logit_diff_delta" in c for c in out["top"])


def test_sweep_rejects_unsupported_ops(model):
    with pytest.raises(ValueError):
        model.sweep(PROMPT, op="edit")


def test_sweep_on_a_single_token_prompt(model):
    out = model.sweep("a", op="uniform")
    assert len(out["tokens"]) == 1
    assert all(math.isfinite(c["kl"]) for row in out["grid"] for c in row)


def test_sweep_respects_a_baseline_intervention(model):
    base = [{"layer": 0, "head": 0, "op": "zero"}]
    out = model.sweep(PROMPT, base_interventions=base)
    assert out["grid"][0][0]["kl"] == 0.0  # already ablated: no further effect


# ---------------- generation ----------------


def test_greedy_generation_is_deterministic_and_bounded(model):
    a = model.generate(PROMPT, max_new_tokens=8)
    b = model.generate(PROMPT, max_new_tokens=8)
    assert a["baseline"] == b["baseline"]
    assert a["baseline"] == a["intervened"]  # no interventions
    assert len(model.tokenizer(a["baseline"]).input_ids) <= 8


def test_generation_diverges_under_a_heavy_intervention(model):
    ivs = [{"layer": l, "head": h, "op": "uniform"} for l in range(model.n_layers)
           for h in range(model.n_heads)]
    out = model.generate(PROMPT, ivs, max_new_tokens=10)
    assert out["baseline"] != out["intervened"]


@pytest.mark.parametrize("n,temp", [(0, 0), (-5, 0), (10_000, 0), (4, float("nan")), (4, 99)])
def test_generation_clamps_its_own_arguments(model, n, temp):
    out = model.generate("Hello", max_new_tokens=n, temperature=temp)
    assert 1 <= out["max_new_tokens"] <= 32
    assert isinstance(out["intervened"], str)


def test_generation_past_the_context_window(model):
    """Prompt at the truncation limit still generates without an index error."""
    out = model.generate(" ".join(["word"] * 200), max_new_tokens=6)
    assert isinstance(out["baseline"], str)


# ---------------- HTTP contract ----------------


def test_health_and_model_info(client):
    assert client.get("/api/health").json()["model_loaded"] is True
    info = client.get("/api/model_info").json()
    assert info["layers"] > 0 and "mask_key" in info["ops"]


def test_analyze_roundtrip_over_http(client):
    r = client.post("/api/analyze", json={
        "text": PROMPT,
        "interventions": [{"layer": 1, "head": 1, "op": "zero"}],
        "targets": [" Mary", " John"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["kl"] >= 0 and "ms" in body and "logit_diff" in body


@pytest.mark.parametrize("payload", [
    {"text": "hi", "interventions": [{"layer": 0, "head": 0, "op": "delete_everything"}]},
    {"text": "hi", "interventions": [{"layer": -1, "head": 0, "op": "zero"}]},
    {"text": "hi", "interventions": [{"layer": 0, "head": 0, "op": "edit",
                                      "cells": [{"q": 0, "k": 0, "value": 7.5}]}]},
    {"text": "hi", "interventions": [{"layer": 0, "head": 0, "op": "zero"}] * 65},
    {"text": "x" * 5000},
    {"text": "hi", "targets": ["a"] * 20},
    {"text": "hi", "interventions": "not-a-list"},
])
def test_malformed_requests_are_rejected_with_422(client, payload):
    assert client.post("/api/analyze", json=payload).status_code == 422


def test_generate_bounds_are_enforced_at_the_edge(client):
    assert client.post("/api/generate", json={"text": "hi", "max_new_tokens": 999}).status_code == 422
    assert client.post("/api/generate", json={"text": "hi", "temperature": -1}).status_code == 422
    ok = client.post("/api/generate", json={"text": "hi", "max_new_tokens": 4})
    assert ok.status_code == 200 and "baseline" in ok.json()


def test_sweep_over_http_rejects_bad_op(client):
    assert client.post("/api/sweep", json={"text": "hi", "op": "edit"}).status_code == 400
    ok = client.post("/api/sweep", json={"text": "hi", "op": "zero"})
    assert ok.status_code == 200 and len(ok.json()["grid"]) > 0


def test_large_response_is_gzipped(client):
    r = client.post(
        "/api/analyze",
        json={"text": PROMPT},
        headers={"Accept-Encoding": "gzip"},
    )
    assert r.headers.get("content-encoding") == "gzip"
