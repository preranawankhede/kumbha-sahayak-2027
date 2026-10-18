
"""
Kumbha Sahayak 2027
Speech recognition + multilingual translation API

Run:
    uvicorn main:app --host 127.0.0.1 --port 8000
"""

import os
import json
import difflib
import tempfile
import traceback

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel

app = FastAPI(title="Kumbha Sahayak 2027 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper language codes -> NLLB language codes
LANGS = {
    "mr": "mar_Deva",
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "gu": "guj_Gujr",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
}

# -------------------------
# 1. Speech recognition
# -------------------------

ASR_MODEL = os.getenv("ASR_MODEL", "base")

asr = WhisperModel(
    ASR_MODEL,
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1,
)

# -------------------------
# 2. Translation model
# -------------------------

MT_NAME = os.getenv(
    "MT_MODEL",
    "facebook/nllb-200-distilled-600M",
)

tokenizer = AutoTokenizer.from_pretrained(MT_NAME)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(MT_NAME)
translation_model.eval()

# -------------------------
# 3. Optional phrase memory
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHRASE_FILE = os.path.join(BASE_DIR, "kumbh_phrases.json")

try:
    with open(PHRASE_FILE, encoding="utf-8") as file:
        PHRASES = json.load(file)
except FileNotFoundError:
    print("Phrase file not found. Using the translation model.")
    PHRASES = {}

def normalize(text):
    return " ".join(
        text.lower().strip(" ?.!,'\"").split()
    )

def phrase_lookup(text, target_language, cutoff=0.88):
    cleaned = normalize(text)

    if not cleaned or not PHRASES:
        return None

    match = difflib.get_close_matches(
        cleaned,
        PHRASES.keys(),
        n=1,
        cutoff=cutoff,
    )

    if not match:
        return None

    entry = PHRASES.get(match[0], {})
    return entry.get(target_language)

# -------------------------
# 4. Translate text
# -------------------------

def translate(text, src, tgt):
    text = text.strip()

    if not text:
        raise ValueError("No text was provided for translation.")

    if src not in LANGS or tgt not in LANGS:
        raise ValueError("Unsupported source or target language.")

    if src == tgt:
        return text, "same-language"

    # Phrase memory is useful for common, verified phrases.
    cached = phrase_lookup(text, tgt)

    if cached:
        return cached, "phrase-memory"

    tokenizer.src_lang = LANGS[src]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    target_id = tokenizer.convert_tokens_to_ids(LANGS[tgt])

    with torch.inference_mode():
        output = translation_model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            num_beams=2,
            max_new_tokens=80,
        )

    result = tokenizer.batch_decode(
        output,
        skip_special_tokens=True,
    )[0].strip()

    if not result:
        raise ValueError("The translation model returned empty text.")

    return result, "nllb"

# -------------------------
# 5. Health check
# -------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "Kumbha Sahayak 2027",
        "speech_recognition": ASR_MODEL,
        "translation_model": MT_NAME,
        "speech_output": "Browser speechSynthesis",
    }

@app.get("/api/languages")
def languages():
    return list(LANGS.keys())

# -------------------------
# 6. Speech -> text -> translation
# -------------------------

@app.post("/api/speech-translate")
async def speech_translate(
    audio: UploadFile = File(...),
    src: str = Form(...),
    tgt: str = Form(...),
):
    if src not in LANGS or tgt not in LANGS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported source or target language.",
        )

    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=422,
            detail="The recorded audio is empty. Please record again.",
        )

    path = None

    try:
        # MediaRecorder in your frontend sends WebM audio.
        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False,
        ) as temp_file:
            temp_file.write(audio_bytes)
            path = temp_file.name

        print("1. Audio received:", len(audio_bytes), "bytes")

        # Stage 1: Speech recognition
        segments, info = asr.transcribe(
            path,
            language=src,
            vad_filter=True,
            beam_size=3,
            condition_on_previous_text=False,
        )

        source_text = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        print("2. Recognized:", source_text)

        if not source_text:
            raise HTTPException(
                status_code=422,
                detail="No speech detected. Speak clearly and try again.",
            )

        # Stage 2: Translation
        translated_text, engine = translate(
            source_text,
            src,
            tgt,
        )

        print("3. Translation:", translated_text)
        print("4. Engine:", engine)

        # JSON response: the browser speaks translated_text.
        return {
            "source_text": source_text,
            "translated": translated_text,
            "engine": engine,
            "source_language": src,
            "target_language": tgt,
        }

    except HTTPException:
        raise

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        )

    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

# -------------------------
# 7. Quick phrase translation
# -------------------------

@app.post("/api/text-translate")
def text_translate(
    text: str = Form(...),
    src: str = Form(...),
    tgt: str = Form(...),
):
    if src not in LANGS or tgt not in LANGS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported source or target language.",
        )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Please provide text to translate.",
        )

    try:
        translated_text, engine = translate(text, src, tgt)

        return {
            "source_text": text,
            "translated": translated_text,
            "engine": engine,
        }

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        )
