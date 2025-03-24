# Connect to your server using the IP address shown (5.161.41.42)
ssh root@5.161.41.42import os
import sys
import json
import asyncio
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
from tenacity import retry, wait_exponential, stop_after_attempt
import aiohttp
import logging

from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    title: str
    summary: str
    content: str
    embedding: List[float]
    chunk_number: int
    metadata: Dict[str, Any] = None

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks, respecting natural boundaries."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        if end < text_length:
            # Try to find the best break point
            chunk = text[start:end]
            
            # Try natural break points in order of preference
            for break_char in ['\n\n', '\n', '. ', ': ']:
                last_break = chunk.rfind(break_char)
                if last_break != -1 and last_break > chunk_size * 0.3:
                    end = start + last_break + len(break_char)
                    break
        
        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position for next chunk, including overlap
        start = max(start + 1, end - overlap)

    return chunks

async def preprocess_query(query: str) -> str:
    """Preprocess the query to improve search quality by extracting key concepts."""
    try:
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{
                "role": "system",
                "content": "Extract 3-5 key technical concepts from this query, separated by commas. Focus on technical terms, framework names, and specific functionality."
            }, {
                "role": "user", 
                "content": query
            }],
            temperature=0
        )
        
        concepts = response.choices[0].message.content.split(',')
        # Combine original query with key concepts
        expanded_query = f"{query} {' '.join(c.strip() for c in concepts)}"
        return expanded_query
    except Exception as e:
        print(f"Error preprocessing query: {e}")
        return query  # Fall back to original query on error

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1500]}..."}
            ],
            response_format={"type": "json_object"},
            timeout=30.0  # Increase timeout
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {str(e)}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        text = text.replace("\n", " ")  # Normalize text
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            timeout=30.0  # Increase timeout
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return [0.0] * 1536  # Return zero vector on failure

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    try:
        # Get title and summary
        extracted = await get_title_and_summary(chunk, url)
        
        # Get embedding
        embedding = await get_embedding(chunk)
        
        # Create metadata
        metadata = {
            "source": "pydantic_ai_docs",
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path
        }
        
        # Create ProcessedChunk object
        processed_chunk = ProcessedChunk(
            url=url,
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk,
            embedding=embedding,
            chunk_number=chunk_number,
            metadata=metadata
        )
        
        # Store the chunk
        await insert_chunk(processed_chunk)
        
        return processed_chunk
        
    except Exception as e:
        print(f"Error processing chunk: {e}")
        raise

async def insert_chunk(chunk: ProcessedChunk):
    """Store processed chunk in Supabase."""
    try:
        # Check if this URL and chunk number combination already exists
        response = await supabase.table('site_pages').select('id').eq('url', chunk.url).eq('chunk_number', chunk.chunk_number).execute()
        
        if response.data:
            print(f"Chunk {chunk.chunk_number} from {chunk.url} already exists, skipping...")
            return
        
        # Insert the chunk
        response = await supabase.table('site_pages').insert({
            'url': chunk.url,
            'title': chunk.title,
            'summary': chunk.summary,
            'content': chunk.content,
            'embedding': chunk.embedding,
            'chunk_number': chunk.chunk_number,
            'metadata': chunk.metadata,
            'source': 'pydantic_ai_docs',
            'created_at': datetime.now(timezone.utc).isoformat()
        }).execute()
        
        print(f"Successfully stored chunk {chunk.chunk_number} from {chunk.url}")
        
    except Exception as e:
        print(f"Error storing chunk: {e}")
        raise

async def get_page_content(url: str) -> str:
    """Get content from a URL using aiohttp."""
    try:
        logging.info(f"Fetching content from {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logging.error(f"Error fetching {url}: {response.status}")
                    return ""
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                # Get text and clean it
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                logging.info(f"Successfully fetched and cleaned content from {url}")
                return text
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return ""

async def process_and_store_document(url: str):
    """Process a document and store it in Supabase."""
    try:
        logging.info(f"Processing {url}")
        content = await get_page_content(url)
        if not content:
            logging.error(f"No content found for {url}")
            return
            
        chunks = chunk_text(content)
        logging.info(f"Split {url} into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {i} from {url}")
                await process_chunk(chunk, i, url)
                logging.info(f"Successfully processed chunk {i} from {url}")
            except Exception as e:
                logging.error(f"Error processing chunk {i} from {url}: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"Error processing document {url}: {str(e)}")

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    urls = get_pydantic_ai_docs_urls()
    chunk_size = 5
    
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        tasks = [process_and_store_document(url) for url in chunk]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    load_dotenv()  # Load environment variables
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crawler.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Log configuration
    logging.info("Starting crawler with configuration:")
    logging.info(f"Supabase URL: {os.getenv('SUPABASE_URL')}")
    logging.info(f"OpenAI API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")

    asyncio.run(main())

# Export the main function and key utilities
__all__ = [
    'crawl_pydantic_ai_docs',
    'ProcessedChunk',
    'chunk_text',
    'insert_chunk',
    'process_chunk',
    'get_embedding',
]

# Rename main to crawl_pydantic_ai_docs for better clarity
crawl_pydantic_ai_docs = main
