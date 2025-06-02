# Short‑Form Info‑Extractor Agent

Give it a public **Instagram Reel / TikTok / YouTube Short URL** ➜  
Get back structured JSON with the place name, location, and genre.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
# Create a .env file with these variables:
OPENAI_API_KEY=sk-...
# optional geocoding
GOOGLE_API_KEY=AIza...

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
├─ vector_manager.py # vector database management
├─ requirements.txt
└─ README.md
```

## Extending

* Swap Whisper for AssemblyAI or Deepgram.  
* Use PaddleOCR for multilingual overlays.  
* Add vision‑only landmarks via Google Vision or CLIP.  

## Deployment

This project is designed to be deployed on Vercel. To deploy:

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Add these environment variables in Vercel:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GOOGLE_API_KEY`: Your Google Maps API key (optional)

The application uses ChromaDB for vector storage, which is a lightweight, in-memory vector database that works well with serverless deployments. The database is automatically persisted to disk in the `chroma_db` directory.
