"""
Query and RAG (Retrieval-Augmented Generation) module for ŚRUTI package.
Handles knowledge retrieval and answer generation using vector search and LLM.
"""

import logging
import re
from typing import Dict, List, Optional, Union

from groq import Groq

from .config import get_config
from .embeddings import load_vectorstore, search_similar_chunks

logger = logging.getLogger(__name__)


def query_knowledge(
    question: str,
    top_k: int = 5,
    persist_dir: Optional[str] = None,
    include_metadata: bool = True,
    temperature: float = 0.3
) -> Dict[str, Union[str, List]]:
    """
    Query the knowledge base using RAG (Retrieval-Augmented Generation).
    
    Args:
        question: Question to ask the knowledge base
        top_k: Number of relevant chunks to retrieve
        persist_dir: Directory where vector database is persisted
        include_metadata: Whether to include detailed source metadata
        temperature: LLM temperature for answer generation
        
    Returns:
        Dictionary with answer and source information
        
    Raises:
        Exception: If query processing fails
    """
    config = get_config()
    config.validate()
    
    try:
        # Step 1: Retrieve relevant chunks
        logger.info(f"Searching for relevant content for question: {question[:100]}...")
        
        similar_chunks = search_similar_chunks(
            query=question,
            k=top_k,
            persist_dir=persist_dir
        )
        
        if not similar_chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or check if the content has been ingested.",
                "sources": [],
                "confidence": "low",
                "chunks_found": 0,
                "avg_similarity": 0.0,
            }
        
        # Step 2: Prepare context for LLM
        context_chunks = []
        sources = []
        
        for i, chunk in enumerate(similar_chunks):
            # Add chunk to context
            chunk_text = chunk["text"]
            metadata = chunk["metadata"]
            
            context_chunks.append(f"[Source {i+1}] {chunk_text}")
            
            # Prepare source information matching Source model:
            # video_id, title, chunk_text, similarity_score, timestamp, chunk_index
            source_info = {
                "video_id": metadata.get("video_id", ""),
                "title": metadata.get("title", "Unknown"),
                "chunk_text": chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text,
                "similarity_score": chunk["similarity_score"],
                "timestamp": metadata.get("timestamp"),
                "chunk_index": metadata.get("chunk_index", i),
            }
            
            sources.append(source_info)
        
        # Step 3: Generate answer using LLM
        logger.info(f"Generating answer using {len(context_chunks)} relevant chunks...")
        
        answer = _generate_answer_with_llm(
            question=question,
            context_chunks=context_chunks,
            temperature=temperature
        )
        
        # Step 4: Determine confidence based on similarity scores
        avg_similarity = sum(chunk["similarity_score"] for chunk in similar_chunks) / len(similar_chunks)
        
        if avg_similarity < 0.5:
            confidence = "low"
        elif avg_similarity < 0.7:
            confidence = "medium"
        else:
            confidence = "high"
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "chunks_found": len(similar_chunks),
            "avg_similarity": avg_similarity,
        }
        
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise


def _generate_answer_with_llm(
    question: str,
    context_chunks: List[str],
    temperature: float = 0.3
) -> str:
    """
    Generate an answer using Groq LLM with retrieved context.
    
    Args:
        question: User's question
        context_chunks: List of relevant text chunks
        temperature: LLM temperature
        
    Returns:
        Generated answer string
    """
    config = get_config()
    
    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        
        # Prepare context
        context = "\n\n".join(context_chunks)
        
        # Build prompt
        prompt = _build_rag_prompt(question, context)
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _get_rag_system_prompt()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=4000,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating answer with LLM: {e}")
        raise


def _get_rag_system_prompt() -> str:
    """Get the system prompt for RAG answer generation."""
    return """You are ŚRUTI (श्रुति), an intelligent knowledge assistant with expertise in multilingual content, particularly Sanskrit and Indic philosophical traditions.

Your role is to answer questions based on the provided context from YouTube video transcripts. You should:

1. **Answer accurately**: Base your response strictly on the provided context
2. **Preserve authenticity**: Maintain original Sanskrit terms, quotes, and cultural nuances
3. **Be comprehensive**: Provide detailed, well-structured answers
4. **Cite sources**: Reference specific sources when making claims
5. **Handle multiple languages**: Seamlessly work with English, Hindi, Sanskrit, and other languages
6. **Acknowledge limitations**: If the context doesn't contain enough information, say so clearly

Guidelines:
- Use Sanskrit terms in Devanagari script when appropriate
- Provide transliterations in parentheses for Sanskrit terms
- Quote directly from sources when relevant
- Structure your answer clearly with proper formatting
- If multiple perspectives exist in the sources, present them fairly
- Don't make claims not supported by the provided context"""


def _build_rag_prompt(question: str, context: str) -> str:
    """Build the RAG prompt for answer generation."""
    return f"""Based on the following context from video transcripts, please answer the question comprehensively and accurately.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based solely on the provided context
- Include relevant quotes and specific details from the sources
- Use proper Sanskrit terms (Devanagari script) when mentioned in the context
- Structure your answer clearly
- If the context doesn't contain sufficient information to answer fully, state this clearly
- Reference which sources ([Source N]) support your claims

ANSWER:"""


def batch_query_knowledge(
    questions: List[str],
    top_k: int = 5,
    persist_dir: Optional[str] = None
) -> List[Dict]:
    """
    Process multiple questions in batch.
    
    Args:
        questions: List of questions to ask
        top_k: Number of relevant chunks to retrieve per question
        persist_dir: Directory where vector database is persisted
        
    Returns:
        List of query results for each question
    """
    results = []
    
    for i, question in enumerate(questions):
        try:
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            result = query_knowledge(
                question=question,
                top_k=top_k,
                persist_dir=persist_dir,
                include_metadata=False  # Keep batch results lighter
            )
            
            result["question"] = question
            result["question_index"] = i
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            results.append({
                "question": question,
                "question_index": i,
                "error": str(e),
                "answer": f"Error processing question: {e}",
                "sources": [],
                "confidence": "error"
            })
    
    return results


def get_related_topics(topic: str, top_k: int = 10, persist_dir: Optional[str] = None) -> List[Dict]:
    """
    Find topics related to the given topic based on semantic similarity.
    
    Args:
        topic: Topic to find related content for
        top_k: Number of related chunks to analyze
        persist_dir: Directory where vector database is persisted
        
    Returns:
        List of RelatedTopic-compatible dicts with topic, relevance_score, video_count
    """
    try:
        # Search for content related to the topic
        similar_chunks = search_similar_chunks(
            query=topic,
            k=top_k,
            persist_dir=persist_dir
        )
        
        if not similar_chunks:
            return []
        
        # Extract topics from chunk metadata and content
        # Track topic -> (best_score, video_ids)
        topic_data: Dict[str, Dict] = {}
        
        for chunk in similar_chunks:
            metadata = chunk["metadata"]
            text = chunk["text"]
            score = chunk["similarity_score"]
            video_id = metadata.get("video_id", "unknown")
            
            # Add video title as a related topic
            if "title" in metadata:
                title = metadata["title"]
                clean_title = re.sub(r'[^\w\s]', '', title).strip()
                if clean_title and len(clean_title) > 10:
                    if clean_title not in topic_data:
                        topic_data[clean_title] = {"score": score, "video_ids": set()}
                    topic_data[clean_title]["video_ids"].add(video_id)
                    topic_data[clean_title]["score"] = max(topic_data[clean_title]["score"], score)
            
            # Extract key phrases from text (simple heuristic)
            phrases = re.findall(r'[A-Z][a-z]+(?: [A-Z][a-z]+)*', text)
            for phrase in phrases[:3]:
                if len(phrase) > 5 and phrase.lower() not in topic.lower():
                    if phrase not in topic_data:
                        topic_data[phrase] = {"score": score, "video_ids": set()}
                    topic_data[phrase]["video_ids"].add(video_id)
                    topic_data[phrase]["score"] = max(topic_data[phrase]["score"], score)
        
        # Convert to RelatedTopic-compatible dicts
        results = []
        for t, data in sorted(topic_data.items(), key=lambda x: x[1]["score"], reverse=True)[:5]:
            results.append({
                "topic": t,
                "relevance_score": data["score"],
                "video_count": len(data["video_ids"]),
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error finding related topics: {e}")
        return []


def search_by_metadata(
    metadata_filter: Dict,
    limit: int = 20,
    persist_dir: Optional[str] = None
) -> List[Dict]:
    """
    Search chunks by metadata filters.
    
    Args:
        metadata_filter: Dictionary of metadata key-value pairs to filter by
        limit: Maximum number of results to return
        persist_dir: Directory where vector database is persisted
        
    Returns:
        List of MetadataSearchResult-compatible dicts
    """
    try:
        vectorstore = load_vectorstore(persist_dir)
        
        # Use a broad query to get documents, then filter by metadata
        results = vectorstore.similarity_search(
            query="",  # Empty query to get diverse results
            k=limit * 3,  # Get more to account for filtering
            filter=metadata_filter
        )
        
        # Format results matching MetadataSearchResult model
        formatted_results = []
        for doc in results[:limit]:
            formatted_results.append({
                "video_id": doc.metadata.get("video_id", ""),
                "title": doc.metadata.get("title", "Unknown"),
                "chunk_text": doc.page_content,
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "metadata": doc.metadata,
            })
        
        logger.info(f"Found {len(formatted_results)} chunks matching metadata filter")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching by metadata: {e}")
        return []