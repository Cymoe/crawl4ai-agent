"""
Agent modules for the crawl4AI system.

This package contains the following agents:
1. DriveWatcher - Monitors Google Drive for changes
2. DocumentProcessor - Processes various document types
3. WebCrawler - Crawls and processes web documentation
4. PydanticAIExpert - Main RAG agent for queries
"""

# Temporarily disabled for initial deployment
# from .gdrive_watcher import DriveWatcher
# from .crawl_gdrive_docs import process_file
from .crawl_pydantic_ai_docs import (
    crawl_pydantic_ai_docs,
    ProcessedChunk,
    chunk_text,
    insert_chunk,
    process_chunk,
    get_embedding,
)
from .pydantic_ai_expert import pydantic_ai_expert

__all__ = [
    # Web Crawler Agent
    'crawl_pydantic_ai_docs',
    'ProcessedChunk',
    'chunk_text',
    'insert_chunk',
    'process_chunk',
    'get_embedding',
    
    # RAG Expert Agent
    'pydantic_ai_expert',
]
