"""FastAPI app: intervention API + static frontend (single deployable unit)."""
import threading
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.model_service import (
    MAX_CELLS,
    MAX_INPUT_TOKENS,
    MAX_INTERVENTIONS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    OPS,
    InterventionModel,
)

@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_safe_warm, daemon=True).start()
    yield


app = FastAPI(title="Attention as a Variable", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
# Attention payloads are large and highly compressible (L*H*T*T rounded floats).
app.add_middleware(GZipMiddleware, minimum_size=1024)

_model: InterventionModel | None = None
_model_lock = threading.Lock()
_load_error: str | None = None


def get_model() -> InterventionModel:
    global _model, _load_error
    with _model_lock:
        if _model is None:
            try:
                _model = InterventionModel()
            except Exception as exc:  # network / disk failure at cold start
                _load_error = f"{type(exc).__name__}: {exc}"
                raise HTTPException(503, f"model unavailable: {_load_error}") from exc
    return _model


def _safe_warm():
    try:
        get_model()
    except Exception:
        pass  # surfaced through /api/health and the next real request


# ---------------- request models ----------------


class Cell(BaseModel):
    q: int = Field(..., ge=0, le=4096)
    k: int = Field(..., ge=0, le=4096)
    value: float = Field(0.0, ge=0.0, le=1.0)


class Intervention(BaseModel):
    layer: int = Field(..., ge=0, le=256)
    head: int = Field(..., ge=0, le=256)
    op: str
    value: float | None = None
    positions: list[int] | None = Field(None, max_length=256)
    cells: list[Cell] | None = Field(None, max_length=MAX_CELLS)
    renormalize: bool = True

    @field_validator("op")
    @classmethod
    def known_op(cls, v: str) -> str:
        if v not in OPS:
            raise ValueError(f"unknown op '{v}'; expected one of {', '.join(OPS)}")
        return v


class AnalyzeRequest(BaseModel):
    text: str = Field("", max_length=4000)
    interventions: list[Intervention] = Field(
        default_factory=list, max_length=MAX_INTERVENTIONS
    )
    targets: list[str] | None = Field(None, max_length=8)
    include_stats: bool = True


class SweepRequest(BaseModel):
    text: str = Field("", max_length=4000)
    op: str = "zero"
    value: float | None = None
    targets: list[str] | None = Field(None, max_length=8)
    interventions: list[Intervention] = Field(
        default_factory=list, max_length=MAX_INTERVENTIONS
    )


class GenerateRequest(BaseModel):
    text: str = Field("", max_length=4000)
    interventions: list[Intervention] = Field(
        default_factory=list, max_length=MAX_INTERVENTIONS
    )
    max_new_tokens: int = Field(12, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


def _ivs(items) -> list[dict]:
    return [i.model_dump() for i in items]


# ---------------- routes ----------------


@app.get("/api/health")
def health():
    return {
        "ok": _load_error is None,
        "model_loaded": _model is not None,
        "error": _load_error,
    }


@app.get("/api/model_info")
def model_info():
    m = get_model()
    return {
        "model": MODEL_ID,
        "layers": m.n_layers,
        "heads": m.n_heads,
        "max_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "ops": list(OPS),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    out = get_model().analyze(
        req.text, _ivs(req.interventions), req.targets, req.include_stats
    )
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/api/sweep")
def sweep(req: SweepRequest):
    t0 = time.perf_counter()
    try:
        out = get_model().sweep(
            req.text, req.op, req.value, req.targets, _ivs(req.interventions)
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/api/generate")
def generate(req: GenerateRequest):
    t0 = time.perf_counter()
    out = get_model().generate(
        req.text, _ivs(req.interventions), req.max_new_tokens, req.temperature
    )
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.exception_handler(Exception)
def unhandled(request: Request, exc: Exception):
    # Never leak a traceback to the browser; keep the UI's error line useful.
    return JSONResponse(
        {"detail": f"internal error: {type(exc).__name__}"}, status_code=500
    )


_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="static")
