"""
ŚRUTI (श्रुति) - Multilingual YouTube Knowledge Extractor

A comprehensive package for extracting, processing, and querying knowledge
from YouTube videos in multiple languages, with special support for Sanskrit
and Indic languages.
"""

__version__ = "0.1.0"
__author__ = "Avyay AI"
__email__ = "team@avyay.ai"

# Lazy imports to avoid dependency issues on package import
__all__ = [
    "transcribe_url",
    "extract_captions", 
    "download_audio",
    "transcribe_audio",
    "extract_knowledge",
    "query_knowledge",
    "get_embeddings",
    "chunk_transcript",
    "store_chunks",
    "load_vectorstore",
]

# Functions will be imported when first accessed
def __getattr__(name):
    """Lazy import of package functions."""
    if name == "transcribe_url":
        from .transcribe import transcribe_url
        return transcribe_url
    elif name == "extract_captions":
        from .transcribe import extract_captions
        return extract_captions
    elif name == "download_audio":
        from .transcribe import download_audio
        return download_audio
    elif name == "transcribe_audio":
        from .transcribe import transcribe_audio
        return transcribe_audio
    elif name == "extract_knowledge":
        from .extract import extract_knowledge
        return extract_knowledge
    elif name == "query_knowledge":
        from .query import query_knowledge
        return query_knowledge
    elif name == "get_embeddings":
        from .embeddings import get_embeddings
        return get_embeddings
    elif name == "chunk_transcript":
        from .embeddings import chunk_transcript
        return chunk_transcript
    elif name == "store_chunks":
        from .embeddings import store_chunks
        return store_chunks
    elif name == "load_vectorstore":
        from .embeddings import load_vectorstore
        return load_vectorstore
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")