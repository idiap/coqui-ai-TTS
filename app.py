from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
from openai import OpenAI
import os
from dotenv import load_dotenv
from TTS.api import TTS
import torch
import uuid
import time  # Zamanlama için eklendi

os.environ["COQUI_TOS_AGREED"] = "1"
load_dotenv()
app = FastAPI()
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
SYSTEM_PROMPT = os.getenv(
    "Sen yardımsever bir sesli asistansın. "
    "Cümleniz metinden sese dönüştürülecek. "
    "En fazla 5 cümle ile cevap verin. "
    "Her zaman *TÜRKÇE* yanıt verin."
)
# ✅ CORS (React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "https://talk-to-tobi.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")
if device == "cuda":
    print(f"GPU adı: {torch.cuda.get_device_name(0)}")

# Modelleri yükle
model = whisper.load_model("turbo")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# Setup OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    total_start = time.time()
    
    # Ses dosyasını geçici olarak kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 1. Transkripsiyon
    transcribe_start = time.time()
    result = model.transcribe(tmp_path, fp16=False, language="tr")
    transcription = result["text"]
    transcribe_time = time.time() - transcribe_start

    # 2. LLM Yanıtı
    llm_start = time.time()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
    )
    llm_response = completion.choices[0].message.content
    llm_time = time.time() - llm_start

    # 3. Ses Üretimi
    tts_start = time.time()
    out_path = f"output_{uuid.uuid4().hex}.wav"
    tts.tts_to_file(
        text=llm_response,
        speaker_wav="clone_voice.wav",
        language="tr",
        file_path=out_path,
    )
    tts_time = time.time() - tts_start

    total_time = time.time() - total_start

    return {
        "user_input": transcription,
        "llm_response": llm_response,
        "audio_url": f"/audio/{os.path.basename(out_path)}",
        "timing": {
            "transcription_seconds": round(transcribe_time, 2),
            "llm_response_seconds": round(llm_time, 2),
            "tts_generation_seconds": round(tts_time, 2),
            "total_seconds": round(total_time, 2)
        },
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "Yalnızca CPU"
        }
    }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "Dosya bulunamadı"}, status_code=404)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)