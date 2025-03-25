"""
Agent modules for the crawl4AI system.

This package contains the following agents:
1. DriveWatcher - Monitors Google Drive for changes
2. DocumentProcessor - Processes various document types
3. WebCrawler - Crawls and processes web documentation
4. PydanticAIExpert - Main RAG agent for queries
"""

from .gdrive_watcher import DriveWatcher
from .crawl_gdrive_docs import process_gdrive_spreadsheet, process_folder, delete_file_from_database
from .crawl_pydantic_ai_docs import chunk_text, crawl_pydantic_ai_docs, ProcessedChunk, insert_chunk, process_chunk, get_embedding
from .pydantic_ai_expert import pydantic_ai_expert

# Note: We don't initialize DriveWatcher here anymore
# It should be initialized in run_watcher.py to avoid multiple instances

__all__ = [
    # Web Crawler Agent
    'crawl_pydantic_ai_docs',
    'chunk_text',
    'insert_chunk',
    'process_chunk',
    'get_embedding',
    
    # RAG Expert Agent
    'pydantic_ai_expert',
    # Drive Watcher Agent
    'DriveWatcher',
    'process_gdrive_spreadsheet',
    'process_folder',
    'delete_file_from_database',
]
