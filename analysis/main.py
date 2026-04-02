from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from analysis.routers import analyze

load_dotenv()

app = FastAPI(
    title="딱 걸렸어! - 허위·과장 광고 의심도 분석 서버",
    description="광고 문구, URL, 이미지를 분석하여 허위·과장 가능성을 탐지합니다.",
    version="0.1.0",
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
    return {"status": "ok", "service": "딱 걸렸어! 분석 서버"}
