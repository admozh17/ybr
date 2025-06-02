# Short‑Form Info‑Extractor Agent

Give it a public **Instagram Reel / TikTok / YouTube Short URL** ➜  
Get back structured JSON with the place name, location, and genre.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export OPENAI_API_KEY=sk-...
# optional geocoding
export GOOGLE_API_KEY=AIza...

# Start the local Qdrant server (required for vector search)
python start_qdrant.py

# Run as CLI tool
python agent.py --url "https://www.instagram.com/reel/C3J.../" --out result.json
jq . result.json

# Run as web app
python web_app.py
# Access at http://0.0.0.0:5000
```

## How it works

1. **Download** the clip with `yt-dlp`.
2. **Transcribe** audio via WhisperX.
3. **OCR** key frames for on‑screen text.
4. **LLM** (GPT‑4o‑mini) parses a unified prompt.
5. **(Optional)** geocode with Google Maps.

## Directory layout

```
├─ agent.py          # orchestrator
├─ extractor.py      # download, ASR, OCR, geocode
├─ llm_parser.py     # prompt & schema
├─ start_qdrant.py   # local vector database server
├─ requirements.txt
└─ README.md
```

## Extending

* Swap Whisper for AssemblyAI or Deepgram.  
* Use PaddleOCR for multilingual overlays.  
* Add vision‑only landmarks via Google Vision or CLIP.  

## Running without Docker

This project now runs without Docker dependencies. Instead of using Docker for the Qdrant vector database, we use a local installation:

1. The `qdrant-server` package is included in `requirements.txt`
2. Use `start_qdrant.py` to run the vector database locally
3. The database files are stored in the `qdrant_data` directory

This makes the project more portable and easier to deploy on platforms like Vercel.
