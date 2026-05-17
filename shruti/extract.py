"""
Knowledge extraction module for ŚRUTI package.
Uses Groq LLM to extract structured knowledge from transcripts.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq

from .config import get_config

logger = logging.getLogger(__name__)


def extract_knowledge(transcript: str, title: str = "") -> Dict[str, Any]:
    """
    Extract structured knowledge from transcript using Groq LLM.
    
    Args:
        transcript: Video transcript text
        title: Video title (optional, provides context)
        
    Returns:
        Dictionary with extracted knowledge components
        
    Raises:
        Exception: If knowledge extraction fails
    """
    config = get_config()
    config.validate()
    
    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        
        # Construct the extraction prompt
        prompt = _build_extraction_prompt(transcript, title)
        
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": _get_system_prompt()
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=8000,
        )
        
        # Parse the JSON response
        content = response.choices[0].message.content
        
        try:
            knowledge = json.loads(content)
            
            # Validate required fields
            required_fields = [
                'summary', 'key_concepts', 'key_quotes', 
                'sanskrit_terms', 'study_questions', 'follow_up_topics'
            ]
            
            for field in required_fields:
                if field not in knowledge:
                    logger.warning(f"Missing field '{field}' in extracted knowledge")
                    knowledge[field] = _get_default_value(field)
            
            return knowledge
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {content[:500]}...")
            
            # Fallback: extract what we can from the response
            return _parse_fallback_response(content, title)
            
    except Exception as e:
        logger.error(f"Error extracting knowledge: {e}")
        raise


def _get_system_prompt() -> str:
    """Get the system prompt for knowledge extraction."""
    return """You are ŚRUTI (श्रुति), an expert knowledge extraction system specialized in multilingual content, particularly Sanskrit and Indic philosophical traditions.

Your task is to analyze video transcripts and extract structured knowledge. You excel at:
- Understanding content in multiple languages (English, Hindi, Sanskrit, etc.)
- Identifying Sanskrit terms and providing accurate transliterations
- Extracting key philosophical and spiritual concepts
- Generating meaningful study questions
- Preserving the authentic voice and quotes from the original content

Always respond with valid JSON in the exact format specified. Preserve original languages for quotes and Sanskrit terms."""


def _build_extraction_prompt(transcript: str, title: str) -> str:
    """Build the extraction prompt for the LLM."""
    context = f"Title: {title}\n\n" if title else ""
    
    return f"""{context}Transcript:
{transcript}

Extract structured knowledge from this content and respond with JSON in exactly this format:

{{
  "summary": "A comprehensive 3-paragraph summary covering the main themes, key insights, and practical takeaways",
  "key_concepts": [
    {{"concept": "Core concept name", "explanation": "Clear explanation of the concept"}},
    {{"concept": "Another concept", "explanation": "Another explanation"}}
  ],
  "key_quotes": [
    "Meaningful quote 1 (preserve original language)",
    "Meaningful quote 2 (preserve original language)"
  ],
  "sanskrit_terms": [
    {{"term": "Sanskrit term in Devanagari", "meaning": "English meaning", "transliteration": "IAST transliteration"}},
    {{"term": "Another term", "meaning": "Another meaning", "transliteration": "Transliteration"}}
  ],
  "study_questions": [
    "Thought-provoking question 1?",
    "Thought-provoking question 2?",
    "... (exactly 10 questions)"
  ],
  "follow_up_topics": [
    "Related topic 1",
    "Related topic 2", 
    "Related topic 3"
  ]
}}

Guidelines:
- Summary should be exactly 3 paragraphs
- Extract 5-10 key concepts maximum
- Preserve original language in quotes (don't translate)
- For Sanskrit terms, use proper Devanagari script and IAST transliteration
- Generate exactly 10 study questions that encourage deep thinking
- Suggest 3-5 follow-up topics for further exploration
- Focus on the most important and meaningful content
- Respond only with valid JSON, no additional text"""


def _get_default_value(field: str) -> Any:
    """Get default value for missing fields."""
    defaults = {
        'summary': "Summary not available due to extraction error.",
        'key_concepts': [],
        'key_quotes': [],
        'sanskrit_terms': [],
        'study_questions': [
            "What are the main themes discussed in this content?",
            "How can these teachings be applied in daily life?",
            "What questions does this content raise for further exploration?"
        ],
        'follow_up_topics': []
    }
    
    return defaults.get(field, [])


def _parse_fallback_response(content: str, title: str) -> Dict[str, Any]:
    """
    Fallback parser when JSON extraction fails.
    Attempts to extract basic information from text response.
    """
    logger.info("Using fallback parsing for malformed JSON response")
    
    # Basic fallback structure
    result = {
        'summary': f"Knowledge extraction from: {title}" if title else "Knowledge extraction completed with partial results.",
        'key_concepts': [],
        'key_quotes': [],
        'sanskrit_terms': [],
        'study_questions': [
            "What are the main themes discussed?",
            "How can these insights be applied practically?",
            "What aspects require further study?"
        ],
        'follow_up_topics': []
    }
    
    # Try to extract some basic information
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for section headers
        if 'summary' in line.lower():
            current_section = 'summary'
        elif 'concept' in line.lower():
            current_section = 'concepts'
        elif 'quote' in line.lower():
            current_section = 'quotes'
        elif 'question' in line.lower():
            current_section = 'questions'
        elif line.startswith('-') or line.startswith('*'):
            # Try to extract list items
            item = line.lstrip('- *').strip()
            if current_section == 'concepts' and item:
                result['key_concepts'].append({
                    'concept': item[:50] + '...' if len(item) > 50 else item,
                    'explanation': 'Extracted from partial response'
                })
            elif current_section == 'quotes' and item:
                result['key_quotes'].append(item)
            elif current_section == 'questions' and item:
                if '?' in item:
                    result['study_questions'].append(item)
    
    return result


def extract_knowledge_batch(transcripts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Extract knowledge from multiple transcripts in batch.
    
    Args:
        transcripts: List of transcript dictionaries with 'text' and optional 'title'
        
    Returns:
        List of extracted knowledge dictionaries
    """
    results = []
    
    for i, transcript_data in enumerate(transcripts):
        try:
            transcript = transcript_data.get('text', '')
            title = transcript_data.get('title', f'Video {i+1}')
            
            knowledge = extract_knowledge(transcript, title)
            knowledge['source_title'] = title
            knowledge['source_index'] = i
            
            results.append(knowledge)
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge from transcript {i}: {e}")
            results.append({
                'error': str(e),
                'source_title': transcript_data.get('title', f'Video {i+1}'),
                'source_index': i
            })
    
    return results