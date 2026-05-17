# ŚRUTI श्रुति — Multilingual Knowledge Extractor

**"That which is heard"** — Transform YouTube videos into structured, searchable knowledge.

Part of the [Avyay (अव्यय) Intelligence Platform](https://avyay.ai).

## What It Does

ŚRUTI extracts structured knowledge from YouTube videos containing **English, Hindi, and Sanskrit** — often mixed within a single video. It transcribes, chunks, embeds, and makes content queryable through a RAG pipeline.

## Features

- 🎙️ **Polyglot Transcription** — Whisper large-v3 with auto language detection (EN/HI/SA)
- 📚 **Knowledge Extraction** — Summaries, key concepts, Sanskrit glossary, study questions
- 🔍 **RAG Search** — Query across your entire video library in natural language
- 🕐 **Timestamp Links** — Every chunk links back to the exact video moment
- 🆓 **Free Tier Stack** — Groq Whisper + Chroma + open-source embeddings

## Quick Start

```bash
pip install avyay-shruti

# Transcribe a video
shruti transcribe "https://youtu.be/55pTFVoclvE"

# Extract knowledge
shruti extract "https://youtu.be/55pTFVoclvE"

# Query your knowledge base
shruti query "What is explained about dharma?"

# Start the API server
shruti serve --port 8000
```

## API

```bash
# Transcribe
curl -X POST http://localhost:8000/api/transcribe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtu.be/55pTFVoclvE"}'

# Query knowledge base
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is dharma?", "top_k": 5}'
```

## Architecture

```
YouTube URL → yt-dlp → Groq Whisper → Chunking → Embeddings → Chroma DB
                                                                    ↓
                                            Query → RAG Retrieval → LLM → Answer
```

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| Audio extraction | yt-dlp | Free |
| Transcription | Groq Whisper API | Free tier |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Free |
| Vector DB | Chroma | Free |
| LLM (RAG) | Groq Llama 3.3 70B | Free tier |
| API Server | FastAPI | Free |
| Deployment | Google Cloud Run | Pay-per-use |

## License

MIT — Built by [Avyay AI](https://avyay.ai)
