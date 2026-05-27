from contextlib import asynccontextmanager
import os
import threading

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analysis.routers import analyze
from analysis.services.ai_analyzer import (
    is_kobert_available,
    is_kobert_loaded,
    warm_kobert_model,
)

load_dotenv()
MODEL_WARMUP_MODE = os.getenv("MODEL_WARMUP_MODE", "sync").strip().lower()


def _build_readiness_payload() -> dict:
    use_kobert = analyze.USE_KOBERT
    return {
        "status": "ok",
        "service": "adcheck-analysis-server",
        "use_kobert": use_kobert,
        "kobert_ready": is_kobert_loaded() if use_kobert else False,
        "kobert_available": is_kobert_available() if use_kobert else False,
    }


def _set_startup_state(
    app: FastAPI,
    *,
    ready: bool,
    warmup_in_progress: bool,
    startup_error: str | None = None,
) -> None:
    app.state.ready = ready
    app.state.warmup_in_progress = warmup_in_progress
    app.state.startup_error = startup_error


def _warm_models_or_raise() -> None:
    if analyze.USE_KOBERT:
        if not is_kobert_available():
            raise RuntimeError("KoBERT model path was not found.")
        if not warm_kobert_model():
            raise RuntimeError("KoBERT model preload failed.")


def _warm_models_in_background(app: FastAPI) -> None:
    try:
        _warm_models_or_raise()
        _set_startup_state(app, ready=True, warmup_in_progress=False)
    except Exception as e:
        _set_startup_state(
            app,
            ready=False,
            warmup_in_progress=False,
            startup_error=str(e),
        )
        print(f"[startup] model warmup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _set_startup_state(app, ready=False, warmup_in_progress=True)
    print(
        "[startup] "
        f"USE_KOBERT={analyze.USE_KOBERT} "
        f"KOBERT_AVAILABLE={is_kobert_available()}"
    )
    if MODEL_WARMUP_MODE == "background":
        threading.Thread(
            target=_warm_models_in_background,
            args=(app,),
            name="model-warmup",
            daemon=True,
        ).start()
    else:
        _warm_models_or_raise()
        _set_startup_state(app, ready=True, warmup_in_progress=False)
    yield


app = FastAPI(
    title="AdCheck Analysis Server",
    description="Analyze ad copy and images for suspicious claims.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/health")
def health_check():
    payload = _build_readiness_payload()
    payload.update(
        {
            "warmup_in_progress": bool(getattr(app.state, "warmup_in_progress", False)),
            "ready": bool(getattr(app.state, "ready", False)),
            "startup_error": getattr(app.state, "startup_error", None),
        }
    )
    return payload


@app.get("/ready")
def readiness_check():
    payload = _build_readiness_payload()
    payload.update(
        {
            "warmup_in_progress": bool(getattr(app.state, "warmup_in_progress", False)),
            "ready": bool(getattr(app.state, "ready", False)),
            "startup_error": getattr(app.state, "startup_error", None),
        }
    )

    if payload["ready"]:
        return payload

    payload["status"] = "not_ready"
    return JSONResponse(status_code=503, content=payload)
