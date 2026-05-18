"""
Embeddings and vector storage module for ŚRUTI package.
Handles text chunking, embedding generation, and Chroma vector database operations.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .config import get_config

logger = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get configured HuggingFace embeddings model.
    
    Returns:
        HuggingFaceEmbeddings instance with multilingual model
    """
    config = get_config()
    
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
        encode_kwargs={'normalize_embeddings': True}
    )


def chunk_transcript(text: str, metadata: Dict) -> List[Document]:
    """
    Chunk transcript text into documents suitable for embedding.
    
    Args:
        text: Transcript text to chunk
        metadata: Metadata to attach to each chunk
        
    Returns:
        List of Document objects with chunked text and metadata
    """
    config = get_config()
    
    # Text splitter with multilingual separators
    # Include Devanagari danda (।) for Sanskrit text
    separators = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        "।",     # Devanagari danda (sentence ender)
        ".",     # English period
        "!",     # Exclamation
        "?",     # Question mark
        ";",     # Semicolon
        ",",     # Comma
        " ",     # Space
        ""       # Character level (last resort)
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )
    
    # Split text into chunks
    chunks = splitter.split_text(text)
    
    # Create Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Add chunk-specific metadata
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": i,
            "chunk_id": str(uuid.uuid4()),
            "text_length": len(chunk),
        })
        
        documents.append(Document(
            page_content=chunk,
            metadata=chunk_metadata
        ))
    
    logger.info(f"Created {len(documents)} chunks from text of length {len(text)}")
    return documents


def store_chunks(chunks: List[Document], persist_dir: Optional[str] = None) -> Chroma:
    """
    Store document chunks in Chroma vector database.
    
    Args:
        chunks: List of Document objects to store
        persist_dir: Directory to persist the vector database (optional)
        
    Returns:
        Chroma vectorstore instance
        
    Raises:
        Exception: If storage fails
    """
    config = get_config()
    
    if persist_dir is None:
        persist_dir = str(config.get_chroma_persist_path())
    
    try:
        # Ensure persist directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Get embeddings model
        embeddings = get_embeddings()
        
        # Create or load vector store
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="shruti_knowledge",
        )
        
        # Add documents
        if chunks:
            # Extract texts and metadatas
            texts = [doc.page_content for doc in chunks]
            metadatas = [doc.metadata for doc in chunks]
            
            # Generate unique IDs for each chunk
            ids = [doc.metadata.get("chunk_id", str(uuid.uuid4())) for doc in chunks]
            
            vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(chunks)} chunks in vector database at {persist_dir}")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error storing chunks in vector database: {e}")
        raise


def load_vectorstore(persist_dir: Optional[str] = None) -> Chroma:
    """
    Load existing Chroma vector database.
    
    Args:
        persist_dir: Directory where vector database is persisted
        
    Returns:
        Chroma vectorstore instance
        
    Raises:
        Exception: If loading fails or database doesn't exist
    """
    config = get_config()
    
    if persist_dir is None:
        persist_dir = str(config.get_chroma_persist_path())
    
    try:
        persist_path = Path(persist_dir)
        if not persist_path.exists():
            raise FileNotFoundError(f"Vector database not found at {persist_dir}")
        
        # Get embeddings model
        embeddings = get_embeddings()
        
        # Load vector store
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="shruti_knowledge",
        )
        
        logger.info(f"Loaded vector database from {persist_dir}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error loading vector database from {persist_dir}: {e}")
        raise


def add_video_to_vectorstore(
    transcript: str, 
    metadata: Dict,
    persist_dir: Optional[str] = None
) -> int:
    """
    Add a complete video transcript to the vector database.
    
    Args:
        transcript: Video transcript text
        metadata: Video metadata (title, video_id, etc.)
        persist_dir: Directory to persist the vector database
        
    Returns:
        Number of chunks stored
        
    Raises:
        Exception: If processing fails
    """
    try:
        # Chunk the transcript
        chunks = chunk_transcript(transcript, metadata)
        
        # Store in vector database
        vectorstore = store_chunks(chunks, persist_dir)
        
        logger.info(f"Added video '{metadata.get('title', 'Unknown')}' "
                   f"({len(chunks)} chunks) to vector database")
        
        return len(chunks)
        
    except Exception as e:
        logger.error(f"Error adding video to vector database: {e}")
        raise


def search_similar_chunks(
    query: str,
    k: int = 5,
    persist_dir: Optional[str] = None,
    filter_metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Search for similar chunks in the vector database.
    
    Args:
        query: Search query text
        k: Number of similar chunks to return
        persist_dir: Directory where vector database is persisted
        filter_metadata: Optional metadata filter
        
    Returns:
        List of similar chunks with metadata and similarity scores
    """
    try:
        # Load vector store
        vectorstore = load_vectorstore(persist_dir)
        
        # Perform similarity search
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_metadata
        )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
            })
        
        logger.info(f"Found {len(formatted_results)} similar chunks for query: {query[:50]}...")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching vector database: {e}")
        raise


def get_database_stats(persist_dir: Optional[str] = None) -> Dict:
    """
    Get statistics about the vector database.
    
    Args:
        persist_dir: Directory where vector database is persisted
        
    Returns:
        Dictionary with database statistics
    """
    try:
        vectorstore = load_vectorstore(persist_dir)
        
        # Get collection info
        collection = vectorstore._collection
        count = collection.count()
        
        # Sample some documents to analyze
        sample_docs = vectorstore.similarity_search("", k=min(10, count))
        
        video_ids = set()
        languages = set()
        
        for doc in sample_docs:
            metadata = doc.metadata
            if "video_id" in metadata:
                video_ids.add(metadata["video_id"])
            if "language" in metadata:
                languages.add(metadata["language"])
        
        return {
            "total_chunks": count,
            "unique_videos": len(video_ids),
            "languages": list(languages),
            "persist_dir": persist_dir or str(get_config().get_chroma_persist_path()),
        }
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)}


def clear_database(persist_dir: Optional[str] = None, confirm: bool = False) -> bool:
    """
    Clear all data from the vector database.
    
    Args:
        persist_dir: Directory where vector database is persisted
        confirm: Must be True to actually clear the database
        
    Returns:
        True if database was cleared, False otherwise
    """
    if not confirm:
        logger.warning("Database clear requires confirm=True")
        return False
    
    config = get_config()
    
    if persist_dir is None:
        persist_dir = str(config.get_chroma_persist_path())
    
    try:
        import shutil
        
        persist_path = Path(persist_dir)
        if persist_path.exists():
            shutil.rmtree(persist_path)
            logger.info(f"Cleared vector database at {persist_dir}")
            return True
        else:
            logger.info(f"No database found at {persist_dir}")
            return False
            
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        return False