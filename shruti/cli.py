"""
Command-line interface for ŚRUTI package.
Provides CLI commands for transcription, extraction, ingestion, and querying.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn

from .config import get_config, validate_config, setup_directories
from .transcribe import transcribe_url
from .extract import extract_knowledge
from .embeddings import add_video_to_vectorstore, get_database_stats
from .query import query_knowledge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config-check', is_flag=True, help='Validate configuration and exit')
@click.pass_context
def cli(ctx, debug, config_check):
    """ŚRUTI (श्रुति) - Multilingual YouTube Knowledge Extractor"""
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    if config_check:
        try:
            validate_config()
            setup_directories()
            click.echo("✅ Configuration is valid")
            sys.exit(0)
        except Exception as e:
            click.echo(f"❌ Configuration error: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.argument('url')
@click.option('--output', '-o', help='Output file (JSON format)')
@click.option('--format', 'output_format', default='text', 
              type=click.Choice(['text', 'json']), 
              help='Output format')
def transcribe(url, output, output_format):
    """Transcribe a YouTube video and print the transcript."""
    
    try:
        # Validate configuration
        validate_config()
        
        click.echo(f"🎬 Transcribing video: {url}")
        
        # Transcribe the video
        result = transcribe_url(url)
        
        if output_format == 'json':
            output_data = result
        else:
            output_data = result['transcript']
        
        # Output to file or stdout
        if output:
            output_path = Path(output)
            if output_format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_data)
            click.echo(f"💾 Saved to: {output_path}")
        else:
            if output_format == 'json':
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(f"\n📝 Transcript ({result['source']}):")
                click.echo(f"Title: {result['title']}")
                click.echo(f"Language: {result['language']}")
                click.echo(f"Duration: {result.get('duration', 'Unknown')} seconds")
                click.echo("\n" + "="*50)
                click.echo(output_data)
        
    except Exception as e:
        click.echo(f"❌ Error transcribing video: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('url')
@click.option('--output', '-o', help='Output file (markdown format)')
def extract(url, output):
    """Extract structured knowledge from a YouTube video."""
    
    try:
        # Validate configuration
        validate_config()
        
        click.echo(f"🎬 Extracting knowledge from: {url}")
        
        # Transcribe the video
        click.echo("📝 Transcribing video...")
        transcript_result = transcribe_url(url)
        
        # Extract knowledge
        click.echo("🧠 Extracting structured knowledge...")
        knowledge = extract_knowledge(
            transcript=transcript_result['transcript'],
            title=transcript_result['title']
        )
        
        # Format as markdown
        markdown = _format_knowledge_as_markdown(knowledge, transcript_result)
        
        # Output to file or stdout
        if output:
            output_path = Path(output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            click.echo(f"💾 Saved knowledge extraction to: {output_path}")
        else:
            click.echo("\n" + "="*60)
            click.echo(markdown)
        
    except Exception as e:
        click.echo(f"❌ Error extracting knowledge: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('url')
@click.option('--persist-dir', help='Vector database directory')
def ingest(url, persist_dir):
    """Ingest a video into the knowledge base (transcribe + chunk + embed + store)."""
    
    try:
        # Validate configuration and setup directories
        validate_config()
        setup_directories()
        
        click.echo(f"🎬 Ingesting video: {url}")
        
        # Transcribe the video
        click.echo("📝 Transcribing video...")
        transcript_result = transcribe_url(url)
        
        # Prepare metadata
        metadata = {
            "video_id": transcript_result["video_id"],
            "title": transcript_result["title"],
            "language": transcript_result["language"],
            "source": transcript_result["source"],
            "duration": transcript_result.get("duration", 0),
            "url": transcript_result["url"],
        }
        
        # Add to vector store
        click.echo("🔍 Chunking and embedding transcript...")
        chunks_stored = add_video_to_vectorstore(
            transcript=transcript_result["transcript"],
            metadata=metadata,
            persist_dir=persist_dir
        )
        
        click.echo(f"✅ Successfully ingested video!")
        click.echo(f"   Title: {transcript_result['title']}")
        click.echo(f"   Video ID: {transcript_result['video_id']}")
        click.echo(f"   Language: {transcript_result['language']}")
        click.echo(f"   Chunks stored: {chunks_stored}")
        
        # Show database stats
        config = get_config()
        db_dir = persist_dir or str(config.get_chroma_persist_path())
        stats = get_database_stats(db_dir)
        click.echo(f"   Total chunks in DB: {stats.get('total_chunks', 'Unknown')}")
        
    except Exception as e:
        click.echo(f"❌ Error ingesting video: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('question')
@click.option('--top-k', default=5, help='Number of relevant chunks to retrieve')
@click.option('--persist-dir', help='Vector database directory')
@click.option('--format', 'output_format', default='text',
              type=click.Choice(['text', 'json']),
              help='Output format')
@click.option('--include-sources', is_flag=True, help='Include detailed source information')
def query(question, top_k, persist_dir, output_format, include_sources):
    """Query the knowledge base using RAG."""
    
    try:
        # Validate configuration
        validate_config()
        
        click.echo(f"🔍 Searching knowledge base for: {question}")
        
        # Query the knowledge base
        result = query_knowledge(
            question=question,
            top_k=top_k,
            persist_dir=persist_dir,
            include_metadata=include_sources
        )
        
        if output_format == 'json':
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Format text output
            click.echo(f"\n🤖 Answer (Confidence: {result['confidence']}):")
            click.echo("="*60)
            click.echo(result['answer'])
            
            if include_sources and result['sources']:
                click.echo(f"\n📚 Sources ({len(result['sources'])} found):")
                click.echo("-"*40)
                
                for i, source in enumerate(result['sources'], 1):
                    click.echo(f"\n[{i}] {source.get('title', 'Unknown Title')}")
                    if 'video_id' in source:
                        click.echo(f"    Video ID: {source['video_id']}")
                        video_url = f"https://youtube.com/watch?v={source['video_id']}"
                        if source.get('timestamp'):
                            video_url += f"&t={int(source['timestamp'])}s"
                        click.echo(f"    URL: {video_url}")
                    click.echo(f"    Similarity: {source['similarity_score']:.3f}")
                    click.echo(f"    Text: {source.get('chunk_text', '')}")
        
    except Exception as e:
        click.echo(f"❌ Error querying knowledge base: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--persist-dir', help='Vector database directory')
def stats(persist_dir):
    """Show knowledge base statistics."""
    
    try:
        config = get_config()
        db_dir = persist_dir or str(config.get_chroma_persist_path())
        
        click.echo(f"📊 Knowledge Base Statistics")
        click.echo("="*40)
        
        stats = get_database_stats(db_dir)
        
        if 'error' in stats:
            click.echo(f"❌ Error: {stats['error']}")
        else:
            click.echo(f"Database location: {stats['persist_dir']}")
            click.echo(f"Total chunks: {stats['total_chunks']}")
            click.echo(f"Unique videos: {stats['unique_videos']}")
            click.echo(f"Languages: {', '.join(stats['languages']) if stats['languages'] else 'Unknown'}")
        
    except Exception as e:
        click.echo(f"❌ Error getting stats: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--port', default=8000, help='Port to run the server on')
@click.option('--host', default='127.0.0.1', help='Host to bind the server to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(port, host, reload):
    """Start the FastAPI server."""
    
    try:
        # Validate configuration
        validate_config()
        setup_directories()
        
        click.echo(f"🚀 Starting ŚRUTI API server on {host}:{port}")
        click.echo(f"📚 API documentation: http://{host}:{port}/docs")
        
        # Import here to avoid circular imports
        from .api import app
        
        uvicorn.run(
            "shruti.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except ImportError:
        click.echo("❌ FastAPI not installed. Install with: pip install 'shruti[api]'", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error starting server: {e}", err=True)
        sys.exit(1)


def _format_knowledge_as_markdown(knowledge: dict, transcript_result: dict) -> str:
    """Format extracted knowledge as markdown."""
    
    markdown = f"""# {transcript_result['title']}

**Video ID:** {transcript_result['video_id']}  
**Language:** {transcript_result['language']}  
**Source:** {transcript_result['source']}  
**Duration:** {transcript_result.get('duration', 'Unknown')} seconds  
**URL:** {transcript_result['url']}

## Summary

{knowledge.get('summary', 'No summary available.')}

## Key Concepts

"""
    
    for concept in knowledge.get('key_concepts', []):
        markdown += f"### {concept.get('concept', 'Unknown Concept')}\n\n"
        markdown += f"{concept.get('explanation', 'No explanation available.')}\n\n"
    
    if knowledge.get('key_quotes'):
        markdown += "## Key Quotes\n\n"
        for quote in knowledge['key_quotes']:
            markdown += f"> {quote}\n\n"
    
    if knowledge.get('sanskrit_terms'):
        markdown += "## Sanskrit Terms\n\n"
        for term in knowledge['sanskrit_terms']:
            sanskrit = term.get('term', '')
            meaning = term.get('meaning', '')
            transliteration = term.get('transliteration', '')
            markdown += f"- **{sanskrit}** ({transliteration}): {meaning}\n"
        markdown += "\n"
    
    if knowledge.get('study_questions'):
        markdown += "## Study Questions\n\n"
        for i, question in enumerate(knowledge['study_questions'], 1):
            markdown += f"{i}. {question}\n"
        markdown += "\n"
    
    if knowledge.get('follow_up_topics'):
        markdown += "## Follow-up Topics\n\n"
        for topic in knowledge['follow_up_topics']:
            markdown += f"- {topic}\n"
    
    return markdown


if __name__ == '__main__':
    cli()