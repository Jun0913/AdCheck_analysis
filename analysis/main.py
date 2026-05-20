from contextlib import asynccontextmanager
import os
import threading

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from analysis.routers import analyze
from analysis.services.ai_analyzer import (
    is_kobert_available,
    is_kobert_loaded,
    warm_kobert_model,
)
from analysis.services.nli_verifier import (
    is_nli_available,
    is_nli_loaded,
    warm_nli_model,
)

load_dotenv()
MODEL_WARMUP_MODE = os.getenv("MODEL_WARMUP_MODE", "sync").strip().lower()


def _warm_models_in_background() -> None:
    try:
        if analyze.USE_KOBERT and is_kobert_available():
            warm_kobert_model()
        if is_nli_available():
            warm_nli_model()
    except Exception as e:
        print(f"[startup] model warmup skipped: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(
        "[startup] "
        f"USE_KOBERT={analyze.USE_KOBERT} "
        f"KOBERT_AVAILABLE={is_kobert_available()} "
        f"NLI_AVAILABLE={is_nli_available()}"
    )
    if MODEL_WARMUP_MODE == "background":
        threading.Thread(target=_warm_models_in_background, name="model-warmup", daemon=True).start()
    else:
        _warm_models_in_background()
    yield


app = FastAPI(
    title="??嫄몃졇?? - ?덉쐞怨쇱옣 愿묎퀬 ?섏떖??遺꾩꽍 ?쒕쾭",
    description="愿묎퀬 臾멸뎄, URL, ?대?吏瑜?遺꾩꽍?섏뿬 ?덉쐞怨쇱옣 媛?μ꽦???쒓났?⑸땲??",
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
    use_kobert = analyze.USE_KOBERT
    return {
        "status": "ok",
        "service": "??嫄몃졇?? 遺꾩꽍 ?쒕쾭",
        "use_kobert": use_kobert,
        "kobert_ready": is_kobert_loaded() if use_kobert else False,
        "kobert_available": is_kobert_available() if use_kobert else False,
        "nli_ready": is_nli_loaded(),
        "nli_available": is_nli_available(),
    }
