"""FastAPI app: intervention API + static frontend (single deployable unit)."""
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.model_service import MAX_INPUT_TOKENS, MODEL_ID, InterventionModel

app = FastAPI(title="Attention as a Variable")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_model: InterventionModel | None = None
_model_lock = threading.Lock()


def get_model() -> InterventionModel:
    global _model
    with _model_lock:
        if _model is None:
            _model = InterventionModel()
    return _model


@app.on_event("startup")
def warm():
    threading.Thread(target=get_model, daemon=True).start()


class Cell(BaseModel):
    q: int
    k: int
    value: float


class Intervention(BaseModel):
    layer: int
    head: int
    op: str  # zero | scale | uniform | self | prev | first | edit
    value: float | None = None
    cells: list[Cell] | None = None
    renormalize: bool = True


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=2000)
    interventions: list[Intervention] = []


@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": _model is not None}


@app.get("/api/model_info")
def model_info():
    m = get_model()
    return {
        "model": MODEL_ID,
        "layers": m.n_layers,
        "heads": m.n_heads,
        "max_tokens": MAX_INPUT_TOKENS,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    return get_model().analyze(
        req.text, [i.model_dump() for i in req.interventions]
    )


_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="static")
