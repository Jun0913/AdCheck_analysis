from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from analysis.routers import analyze
from analysis.services.ai_analyzer import _load_model as load_kobert_model
from analysis.services.nli_verifier import _load_nli_model

load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI):
    use_kobert = analyze.USE_KOBERT
    kobert_ready = load_kobert_model() if use_kobert else False
    print(f"[startup] USE_KOBERT={use_kobert} KoBERT_READY={kobert_ready}")

    # NLI is conditional, but validating it at startup makes stage failures visible.
    nli_ready = _load_nli_model()
    print(f"[startup] NLI_READY={nli_ready}")
    yield


app = FastAPI(
    title="딱 걸렸어! - 허위·과장 광고 의심도 분석 서버",
    description="광고 문구, URL, 이미지를 분석하여 허위·과장 가능성을 탐지합니다.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정 (Spring Boot 백엔드 및 프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 실제 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)


@app.get("/health")
def health_check():
    use_kobert = analyze.USE_KOBERT
    kobert_ready = load_kobert_model() if use_kobert else False
    nli_ready = _load_nli_model()
    return {
        "status": "ok",
        "service": "딱 걸렸어! 분석 서버",
        "use_kobert": use_kobert,
        "kobert_ready": kobert_ready,
        "nli_ready": nli_ready,
    }
