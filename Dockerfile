FROM python:3.11-slim

# Install system dependencies (ffmpeg for yt-dlp audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY shruti/ shruti/

# Install CPU-only PyTorch first (avoids ~2GB CUDA download), then the package
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Pre-download the embedding model at build time so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Create volume for Chroma persistence
VOLUME /data/knowledge_base
ENV CHROMA_PERSIST_DIR=/data/knowledge_base

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "shruti.api:app", "--host", "0.0.0.0", "--port", "8000"]