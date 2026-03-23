# realtime_voice_agent.py

# =========================
# 1. 라이브러리 import
# =========================
import os              # 환경변수 사용
import json            # JSON 데이터 처리
import base64          # 오디오 데이터를 base64로 인코딩/디코딩
import asyncio         # 비동기 처리
import websockets      # OpenAI Realtime API 연결용 (WebSocket 클라이언트)
from fastapi import FastAPI, WebSocket  # FastAPI + WebSocket 서버

# =========================
# 2. 환경 변수 로드 (.env)
# =========================
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일 읽어서 환경변수로 등록
except ImportError:
    pass

# OpenAI API 키 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# =========================
# 3. 설정값
# =========================
VOICE = "Coral"   # 사용할 음성 스타일
PCM_SR = 24000    # 샘플레이트 (Hz)
PORT = 5050       # 서버 포트

# FastAPI 앱 생성
app = FastAPI()

# =========================
# 4. WebSocket 엔드포인트
# =========================
@app.websocket("/voice")
async def voice_bridge(ws: WebSocket) -> None:
    """
    브라우저 <-> OpenAI Realtime API 사이를 중계하는 핵심 함수
    """

    # 클라이언트(Web 브라우저) 연결 수락
    await ws.accept()

    # =========================
    # 5. OpenAI Realtime API 연결
    # =========================
    openai_ws = await websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        },
        max_size=None,
        max_queue=None  # 메시지 크기/큐 제한 제거 (데모용)
    )

    # =========================
    # 6. 세션 초기화 (AI 설정)
    # =========================
    await openai_ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},  # 음성 감지 (말 시작/끝)
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "voice": VOICE,
            "modalities": ["audio"],  # 음성만 사용
            "instructions": "당신은 간결한 AI 에이전트입니다..."
        }
    }))

    # =========================
    # 7. 상태 변수
    # =========================
    last_assistant_item = None  # 현재 AI 응답 추적
    latest_pcm_ts = 0           # 음성 타임스탬프(ms)
    pending_marks = []          # (현재 코드에서는 미사용)

    # =========================
    # 8. 클라이언트 → OpenAI (음성 업로드)
    # =========================
    async def from_client() -> None:
        """
        브라우저에서 받은 마이크 음성을 OpenAI로 전달
        """
        nonlocal latest_pcm_ts

        async for msg in ws.iter_text():
            data = json.loads(msg)

            # base64 → PCM raw 데이터 변환
            pcm = base64.b64decode(data["audio"])

            # 오디오 길이를 ms로 계산 (샘플레이트 기반)
            latest_pcm_ts += int(len(pcm) / (PCM_SR * 2) * 1000)

            # OpenAI로 오디오 전송
            await openai_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }))

    # =========================
    # 9. OpenAI → 클라이언트 (음성 응답)
    # =========================
    async def to_client() -> None:
        """
        OpenAI에서 받은 음성 응답을 브라우저로 전달
        + 인터럽션 처리
        """
        nonlocal last_assistant_item, pending_marks

        async for row in openai_ws:
            msg = json.loads(row)

            # -------------------------
            # 9-1. AI 음성 응답 전달
            # -------------------------
            if msg["type"] == "response.audio.delta":
                pcm = base64.b64decode(msg["delta"])

                # 다시 base64로 인코딩해서 클라이언트로 전송
                await ws.send_json({
                    "audio": base64.b64encode(pcm).decode("ascii")
                })

                # 현재 응답 ID 저장
                last_assistant_item = msg.get("item")

            # -------------------------
            # 9-2. 인터럽션 처리 (중요)
            # -------------------------
            # 사용자가 말하기 시작하면
            if msg["type"] == "input_audio_buffer.speech_started" and last_assistant_item:

                # AI 발화 강제 중단
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.truncate",
                    "item_id": last_assistant_item,
                    "content_index": 0,
                    "audio_end_ms": 0
                }))

                # 상태 초기화
                last_assistant_item = None
                pending_marks.clear()

    # =========================
    # 10. 양방향 통신 실행
    # =========================
    try:
        # 클라이언트→AI / AI→클라이언트 동시에 실행
        await asyncio.gather(from_client(), to_client())

    finally:
        # 연결 종료 처리
        await openai_ws.close()
        await ws.close()

# =========================
# 11. 서버 실행
# =========================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("\n실시간 음성 에이전트 서버가 시작됩니다...")
    print("\n브라우저에서 index.html을 실행하세요!")
    print("="*60 + "\n")

    # FastAPI 앱 실행
    uvicorn.run(
        "realtime_voice_agent:app",
        host="0.0.0.0",
        port=PORT
    )