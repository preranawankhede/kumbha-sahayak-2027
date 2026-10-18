# Kumbha Sahayak 2027

**A speech-to-speech language translator for pilgrims at Kumbh Mela 2027, Nashik.**

Speak in your language, hear the translation in another. Kumbha Sahayak chains three open AI models behind our own REST API and adds a human-verified Kumbh phrase memory for directions, help and emergencies.

> Status: working prototype. Dataset collection and fine-tuning on Kumbh-specific data are in progress (see [Roadmap](#roadmap)).

---

## Why this project

Millions of pilgrims from all over India visit Nashik during Kumbh Mela. Many speak only their own language, are elderly, and cannot easily type in a translation app, while volunteers, police and medical staff mostly speak Marathi, Hindi or English. Voice is the natural interface, so the app lets a person speak and hear the answer.

## Features

- Speech to speech translation between Marathi, Hindi, English and Gujarati (Tamil, Telugu, Kannada, Bengali and Punjabi are configured and can be enabled after quality testing)
- Kumbh phrase memory: fuzzy-matched, human-checked answers for common phrases (instant and safer than AI for emergencies)
- Own REST API with automatic interactive documentation at `/docs`
- Mobile-friendly web frontend with a microphone button and quick-phrase buttons
- Models are loaded once at startup and reused for every request
- Pluggable models: point `ASR_MODEL` and `MT_MODEL` at your own fine-tuned versions

## How it works

```
Browser (mic) --audio + src + tgt--> FastAPI backend
                                       1. Whisper ASR      audio -> text
                                       2. Phrase memory    known Kumbh phrase? -> stored translation
                                       3. NLLB-200         text -> translated text
                                       4. MMS-TTS (VITS)   translated text -> WAV audio
Browser (speaker) <--audio + texts------
```

| Stage | Model | Algorithm |
|---|---|---|
| Speech recognition | Whisper (via faster-whisper) | Log-Mel spectrogram, Transformer encoder-decoder, beam search (5) |
| Phrase memory | JSON + difflib | Fuzzy string similarity, cutoff 0.82 |
| Translation | NLLB-200 distilled 600M | Subword tokenization, self-attention, cross-attention, beam search (4) |
| Speech synthesis | Meta MMS-TTS (VITS) | Text encoder, duration/pitch predictor, flow decoder + HiFi-GAN vocoder |

## Tech stack

Python, FastAPI, Uvicorn, PyTorch, Hugging Face Transformers, faster-whisper, soundfile, HTML, CSS, JavaScript (MediaRecorder API).

## Project structure

```
kumbha-sahayak-2027/
├── backend/
│   ├── main.py              # FastAPI app: ASR -> phrase memory/MT -> TTS
│   ├── kumbh_phrases.json   # verified Kumbh phrases
│   └── requirements.txt
├── frontend/
│   └── index.html           # web app
├── .gitignore
└── README.md
```

## Installation (Windows PowerShell)

Requirements: Python 3.10 to 3.12 recommended, about 5 GB free disk space (models download on first run), internet connection for the first run.

```powershell
git clone https://github.com/<your-username>/kumbha-sahayak-2027.git
cd kumbha-sahayak-2027\backend

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS use `source .venv/bin/activate`.

## Running

**1. Start the backend** (keep this window open; do not press Ctrl+C in it):

```powershell
cd backend
python -m uvicorn main:app --port 8000
```

Wait for `Application startup complete`. The first start downloads the Whisper and NLLB models, and the first request into each language downloads that language's voice model.

Check it: http://127.0.0.1:8000/api/health

**2. Start the frontend** (a second terminal):

```powershell
cd frontend
python -m http.server 5500
```

Open http://127.0.0.1:5500/index.html. Microphone access works on `localhost`/`127.0.0.1` or over HTTPS.

If the page cannot reach the backend, check the `API` constant near the top of the script in `index.html` (`http://127.0.0.1:8000/api`).

## API reference

Interactive docs: http://127.0.0.1:8000/docs

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server status |
| `/api/languages` | GET | Supported language codes |
| `/api/speech-translate` | POST (multipart form) | `audio` file, `src`, `tgt` -> WAV audio. The recognized and translated text are returned in the `X-Source-Text` and `X-Translated-Text` headers, and `X-Engine` shows `phrase-memory` or `nmt` |
| `/api/text-translate` | POST (form) | `text`, `src`, `tgt` -> `{"translated": "...", "engine": "..."}` |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/text-translate \
  -F "text=Where is the toilet?" -F "src=en" -F "tgt=mr"
```

Language codes: `mr` Marathi, `hi` Hindi, `en` English, `gu` Gujarati, `ta` Tamil, `te` Telugu, `kn` Kannada, `bn` Bengali, `pa` Punjabi.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `ASR_MODEL` | `small` | Whisper model name or path to your fine-tuned CTranslate2 model |
| `MT_MODEL` | `facebook/nllb-200-distilled-600M` | Translation model name or path to your fine-tuned model |

PowerShell example:

```powershell
$env:MT_MODEL = "C:\models\kumbha-mt-final"
python -m uvicorn main:app --port 8000
```

## Adding phrases

Edit `backend/kumbh_phrases.json`. Keys are lower-case English phrases without punctuation, values map language codes to the verified translation:

```json
{
  "where is ramkund": {"mr": "रामकुंड कुठे आहे?", "hi": "रामकुंड कहाँ है?", "gu": "રામકુંડ ક્યાં છે?"}
}
```

Phrases for medical, police and emergency use should be checked by native speakers.

## Dataset (for fine-tuning)

Translation data (CSV):

```
id,category,en,hi,mr,gu
1,direction,"Where is Ramkund?","रामकुंड कहाँ है?","रामकुंड कुठे आहे?","રામકુંડ ક્યાં છે?"
```

Speech data: 16 kHz mono WAV files plus a transcript file. Split 80/10/10 (train/validation/test) and never let test sentences appear in training. Obtain consent from every voice contributor.

Public data used to keep general ability and for evaluation: Samanantar and BPCC (AI4Bharat), Common Voice, IndicVoices, Kathbath, FLORES-200.

## Evaluation

| Component | Metric |
|---|---|
| ASR | Word error rate (WER) on clean and noisy audio |
| Translation | chrF, BLEU, and review by native speakers |
| TTS | Listening test by native speakers |
| System | End-to-end latency on real phones and weak networks |

## Roadmap

- [x] Working pipeline: ASR, phrase memory, translation, TTS
- [x] FastAPI backend with four endpoints and web frontend
- [ ] Collect Kumbh sentence pairs and noisy voice recordings
- [ ] Fine-tune NLLB (translation) and Whisper (ASR) and report improvement over the base models
- [ ] API keys, rate limiting, logging and error handling
- [ ] Installable web app (PWA) and offline phrase mode
- [ ] Cloud deployment with HTTPS and load testing

## Known limitations

- Translations can be wrong, especially for names, numbers and rare words. Do not rely on them alone for medical or emergency situations.
- Pretrained models are not tuned for crowded outdoor audio until fine-tuning is done.
- MMS voices are understandable but not perfectly natural.
- Three models need a capable machine; a GPU is recommended for production.
- Errors accumulate in a cascade: a recognition mistake carries into translation and speech.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn` is not recognized | Run it as `python -m uvicorn main:app --port 8000` inside the activated venv |
| `pip` finds no matching version | Use a supported Python version (3.10 to 3.12) and check your internet connection |
| Browser shows connection refused | The backend or the frontend server is not running, or use `127.0.0.1` instead of `localhost` |
| Page shows a CORS error with status 500 | The server crashed; read the traceback in the backend terminal |
| Do not press Ctrl+C to copy text | It stops the server; right-click to copy instead |

## Privacy

Uploaded audio is saved to a temporary file only for processing and deleted afterwards. Audio is not stored by default.

## Acknowledgements

- OpenAI Whisper and the faster-whisper project
- Meta AI: NLLB-200 and MMS-TTS
- AI4Bharat: Samanantar, BPCC, IndicVoices, Kathbath
- Hugging Face Transformers, FastAPI

Please check the license of each model and dataset before any commercial use.

## References

- Vaswani et al., Attention Is All You Need, 2017
- Radford et al., Robust Speech Recognition via Large-Scale Weak Supervision (Whisper), 2022
- NLLB Team, No Language Left Behind, 2022
- Kim et al., VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech, 2021
- Pratap et al., Scaling Speech Technology to 1,000+ Languages (MMS), 2023

## License

Choose a license for your own code (for example MIT) and add a `LICENSE` file. Pretrained models keep their own licenses.

## Author

Your name, college and department.
