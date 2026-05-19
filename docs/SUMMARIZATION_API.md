# ŚRUTI V3.0 — Advanced Video Summarization API

## Overview

ŚRUTI V3.0 introduces advanced video summarization capabilities combining chapter detection, key insights extraction, sentiment analysis, and multi-format exports.

## New Endpoints

### `POST /v3/summarize`

Advanced video summarization with chapter detection and key insights.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "max_insights": 50,
  "min_insight_importance": 0.3,
  "enable_visual": true,
  "enable_speakers": true,
  "max_chapters": 15,
  "export_format": "markdown"
}
```

**Response includes:**
- `executive_summary` — Concise summary of the video
- `chapters` — Enhanced chapters with sentiment, visual content flags, confidence scores
- `key_insights` — Extracted insights (key points, action items, quotes, statistics, etc.)
- `action_items` — Structured action items with priority levels
- `top_quotes` — Most quotable moments
- `sentiment_arc` — Sentiment analysis across the video timeline
- `youtube_chapters` — Ready-to-paste YouTube chapter timestamps
- `quality_report` — Quality metrics with grade (A-F)
- `export` — Optional pre-formatted export (markdown/json/youtube/srt/vtt)

### `POST /v3/insights/search`

Search across all processed video insights.

**Request:**
```json
{
  "query": "architecture",
  "insight_type": "key_point",
  "min_importance": 0.5,
  "video_id": "optional-filter",
  "limit": 50
}
```

### `POST /v3/chapters/export`

Export chapter timestamps in various formats.

**Query params:** `format=youtube|srt|vtt|ffmpeg`

## Architecture

### Module Structure

```
shruti/
├── insights.py      # Key insights extraction engine
├── summarize.py     # Orchestration: chapter detection + insights + quality
├── exports.py       # Export formats: markdown, JSON, SRT, VTT, ffmpeg
├── segments.py      # Topic segmentation & chapter detection (existing)
├── visual.py        # Visual analysis: OCR, scene detection (existing)
└── speakers.py      # Speaker diarization (existing)
```

### Pipeline Flow

```
Transcript + Segments
        │
        ├──► Enhanced Chapter Detection
        │       ├── Topic segmentation (embedding-based)
        │       ├── Scene change alignment
        │       ├── Speaker change alignment
        │       └── Smart buffer zones
        │
        ├──► Key Insights Extraction
        │       ├── Emphasis detection (key phrases, repetition)
        │       ├── Action item identification
        │       ├── Question detection
        │       ├── Statistics/data extraction
        │       ├── Quote extraction
        │       ├── Definition detection
        │       └── Conclusion detection
        │
        ├──► Sentiment Analysis
        │       ├── Lexicon-based scoring (positive/negative words)
        │       ├── Sliding window arc computation
        │       └── Per-chapter sentiment assignment
        │
        └──► Quality Scoring
                ├── Chapter coverage
                ├── Insight density & diversity
                ├── Confidence metrics
                └── Grade assignment (A-F)
```

## Insight Types

| Type | Description | Example Pattern |
|------|-------------|-----------------|
| `key_point` | Important statements | "The most important thing..." |
| `action_item` | Actionable recommendations | "You should...", "Make sure..." |
| `question` | Questions (rhetorical/real) | "What if...?", "Have you ever...?" |
| `quote` | Quotable highlights | Impactful standalone statements |
| `definition` | Definitions/explanations | "X is defined as...", "In other words..." |
| `statistic` | Data points/statistics | "78% of...", "According to research..." |
| `emphasis` | Emphasized/repeated points | "Remember...", repeated phrases |
| `transition` | Topic transitions | "Now let's move on to..." |
| `conclusion` | Conclusions/summaries | "In conclusion...", "To sum up..." |

## Export Formats

- **Markdown** — Full formatted report with all sections
- **JSON** — Structured data for programmatic use
- **YouTube** — Copy-paste chapter timestamps for descriptions
- **SRT** — Subtitle format for chapter overlay
- **WebVTT** — Web-compatible subtitle format
- **ffmpeg** — Metadata format for chapter embedding in video files

## Quality Scoring

Quality is scored 0-100% with a letter grade:

| Metric | Weight | Description |
|--------|--------|-------------|
| Chapter Coverage | 25% | % of video duration covered by chapters |
| Insight Type Diversity | 20% | Variety of insight types detected |
| Insight Confidence | 20% | Average pattern match confidence |
| Sentiment Coverage | 15% | Completeness of sentiment analysis |
| Insight Importance | 20% | Average importance of insights |

## Python API Usage

```python
from shruti.summarize import summarize_video, SummarizationConfig
from shruti.exports import export_summary_markdown, export_summary_json

# Summarize
result = summarize_video(
    transcript="...",
    video_id="my-video",
    title="My Video",
    segments=[{"start": 0, "end": 10, "text": "..."}],
    total_duration=360.0,
)

# Access results
print(result.summary.executive_summary)
print(f"Quality: {result.quality_report['grade']}")
for ch in result.enhanced_chapters:
    print(f"  {ch.title} ({ch.start_time:.0f}s - {ch.end_time:.0f}s)")

# Export
markdown = export_summary_markdown(result)
json_data = export_summary_json(result)
```

## Insights Database

Processed video insights are automatically saved to a searchable JSON database:

```python
from shruti.exports import save_insights_to_db, search_insights

# Save
save_insights_to_db(result, "/path/to/db")

# Search
results = search_insights(
    "/path/to/db",
    query="security",
    insight_type="action_item",
    min_importance=0.5,
)
```
