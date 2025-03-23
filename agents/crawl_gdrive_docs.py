import os
import json
import asyncio
import tempfile
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import io
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from dotenv import load_dotenv
import openpyxl

from openai import AsyncOpenAI
from supabase import create_client, Client

# Import our existing processing functions
from .crawl_pydantic_ai_docs import (
    ProcessedChunk,
    chunk_text,
    insert_chunk
)

# Load environment variables
load_dotenv()

# Initialize OpenAI and Supabase clients (reusing from existing code)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def build_service():
    """Initialize and return the Google Drive service."""
    creds = None
    token_path = 'credentials/token.json'
    
    # Check if we have valid credentials
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If no valid credentials, let's authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials/client_secret.json', 
                SCOPES,
                redirect_uri='http://localhost:8080'
            )
            creds = flow.run_local_server(port=8080)
        
        # Save the credentials for future use
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

async def process_gdrive_spreadsheet(file_id: str, mime_type: str) -> List[ProcessedChunk]:
    """Process a Google Drive spreadsheet."""
    chunks = []
    service = build_service()
    
    try:
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            # Download the spreadsheet
            print(f"Processing Google Sheet: {file_id}")
            spreadsheet = service.files().get(fileId=file_id).execute()
            
            # Get the spreadsheet as CSV
            request = service.files().export_media(fileId=file_id, mimeType='text/csv')
            content = request.execute()
            
            # Save to temp file
            print("Successfully downloaded spreadsheet")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            temp_file.write(content)
            temp_file.close()
            print("Saved to temp file")
            
            # Read with pandas
            df = pd.read_csv(temp_file.name)
            os.unlink(temp_file.name)
            
            # Process each sheet
            print(f"Found sheets: {[spreadsheet.get('name', 'Sheet1')]}")
            sheet_name = spreadsheet.get('name', 'Sheet1')
            print(f"\nProcessing sheet: {sheet_name}")
            
            # Get sheet info
            print(f"Sheet {sheet_name} has {len(df)} rows and columns: {df.columns.tolist()}")
            
            # Process the dataframe
            chunk = await process_dataframe(df, sheet_name)
            chunks.append(chunk)
            
            print(f"Processed {len(chunks)} sheets")
            
        return chunks
        
    except Exception as e:
        print(f"Error processing spreadsheet: {e}")
        return []

async def process_dataframe(df: pd.DataFrame, file_name: str) -> ProcessedChunk:
    """Process a dataframe into a chunk."""
    print("\n=== Processing DataFrame ===")
    print(f"Sheet: {file_name}")
    print("Raw Data Sample:")
    print(df.head())
    
    # Determine the type of data based on file name
    data_type = 'unknown'
    if 'revenue_metrics' in file_name.lower():
        data_type = 'revenue_metrics'
    elif 'transaction' in file_name.lower():
        data_type = 'transactions'
    elif 'service_package' in file_name.lower():
        data_type = 'service_packages'
    elif 'geographical' in file_name.lower():
        data_type = 'geographical'
    elif 'operational' in file_name.lower():
        data_type = 'operations'
    elif 'customer' in file_name.lower():
        data_type = 'customers'
    
    # Convert DataFrame to text format
    content = "# Raw Data\n\n## Data Records:\n\n"
    for _, row in df.iterrows():
        content += "Record:\n"
        for col in df.columns:
            content += f"- {col}: {row[col]}\n"
        content += "\n"
    
    # Add file ID to content
    content += f"# Data Analysis\nFile ID: {file_name[:20]}..."
    
    print("\nContent Preview:")
    print(content[:500] + "..." if len(content) > 500 else content)
    
    # Create metadata
    metadata = {
        'source': 'gdrive',
        'file_name': file_name,
        'type': data_type,
        'file_type': 'spreadsheet'
    }
    
    # Generate summary
    summary = f"Data from {file_name} with {len(df)} rows and {len(df.columns)} columns"
    
    # Print chunk metadata
    print("\nChunk Metadata:")
    print(f"Title: {file_name}")
    print(f"URL: gdrive://{file_name}")
    print(f"Metadata: {metadata}")
    
    return ProcessedChunk(
        title=file_name,
        url=f"gdrive://{file_name}",
        content=content,
        metadata=metadata,
        summary=summary,
        chunk_number=0,
        embedding=None  # Will be added later
    )

async def process_file(service, file: Dict[str, str]):
    """Process a single file from Google Drive."""
    file_id = file['id']
    mime_type = file['mimeType']
    name = file['name']
    
    print(f"- {name} ({mime_type})")
    
    try:
        chunks = []
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            chunks = await process_gdrive_spreadsheet(file_id, mime_type)
        elif mime_type == 'text/csv':
            # Handle CSV files
            request = service.files().get_media(fileId=file_id)
            content = request.execute()
            
            # Read CSV content
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            chunk = await process_dataframe(df, name)
            chunks = [chunk]
            
        for chunk in chunks:
            print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
            print(f"Processed chunk {chunk.chunk_number} for file {name}")
            
            # Insert into database
            await insert_chunk(chunk)
            
    except Exception as e:
        print(f"Error processing file {name}: {e}")

async def process_folder(folder_id: str):
    """Process all supported files in a Google Drive folder."""
    service = build_service()
    
    try:
        # First, verify we can access the folder
        folder = service.files().get(fileId=folder_id).execute()
        print(f"Found folder: {folder['name']}")
        
        # List all supported files in the folder
        supported_types = [
            "application/vnd.google-apps.document",
            "text/csv",
            "application/vnd.google-apps.spreadsheet",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel"
        ]
        query = f"'{folder_id}' in parents and ("
        query += " or ".join([f"mimeType='{mime}'" for mime in supported_types])
        query += ")"
        
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType)"
        ).execute()
        
        files = results.get('files', [])
        if not files:
            print('No supported files found in the specified folder.')
            return
        
        print(f"\nFound {len(files)} files to process:")
        for file in files:
            print(f"- {file['name']} ({file['mimeType']})")
        
        # Process all files in parallel
        tasks = [process_file(service, file) for file in files]
        await asyncio.gather(*tasks)
        
    except HttpError as error:
        print(f"Error accessing folder {folder_id}: {error}")

async def clear_gdrive_data():
    """Clear existing Google Drive data from the database."""
    try:
        result = supabase.table('site_pages').delete().like('url', 'gdrive://%').execute()
        print(f"Cleared {len(result.data)} existing Google Drive entries")
    except Exception as e:
        print(f"Error clearing data: {e}")

async def clear_all_data():
    """Clear all data from the database."""
    try:
        print("Clearing all data from database...")
        response = supabase.table('site_pages').delete().neq('url', '').execute()
        print(f"Deleted {len(response.data)} documents")
    except Exception as e:
        print(f"Error clearing data: {e}")
        raise e

async def debug_database():
    """Print all documents in the database."""
    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
    all_docs = supabase.table('site_pages').select('*').execute()
    
    print("\nAll documents in database:")
    for doc in all_docs.data:
        print(f"\nDocument:")
        print(f"URL: {doc.get('url')}")
        print(f"Title: {doc.get('title')}")
        print(f"Content preview: {doc.get('content')[:200]}...")
        print(f"Metadata: {doc.get('metadata', {})}")
        print(f"Embedding present: {'Yes' if doc.get('embedding') else 'No'}")

async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI API."""
    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    text = text.replace("\n", " ")
    result = await openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return result.data[0].embedding

async def get_title_and_summary(text: str, url: str) -> Tuple[str, str]:
    """Get a title and summary for a chunk of text."""
    openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # For Google Drive files, extract the file name from the URL
    if url.startswith('gdrive://'):
        if 'Revenue_Metrics_Data' in url:
            return 'Revenue Metrics Analysis', 'Analysis of revenue metrics data including MRR, customers, and other KPIs'
        elif 'service_package_data' in url:
            return 'Service Package Analysis', 'Analysis of service package offerings and their details'
        elif 'transaction_data' in url:
            return 'Transaction Data Analysis', 'Analysis of customer transaction data'
    
    # For other files, generate a title and summary
    prompt = f"Please provide a title and one-sentence summary for this text:\n\n{text[:1000]}..."
    response = await openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = response.choices[0].message.content
    try:
        title, summary = content.split('\n', 1)
        title = title.replace('Title: ', '').strip()
        summary = summary.replace('Summary: ', '').strip()
    except:
        title = url
        summary = content.strip()
    
    return title, summary

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a chunk into Supabase."""
    try:
        print(f"\n=== Inserting Chunk ===")
        print(f"Title: {chunk.title}")
        print(f"URL: {chunk.url}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content Preview: {chunk.content[:200]}...")
        
        # First clear any existing chunks for this URL
        response = supabase.table('site_pages').delete().eq('url', chunk.url).execute()
        print(f"Deleted {len(response.data)} existing chunks")
        
        # Insert the new chunk
        response = supabase.table('site_pages').insert({
            'url': chunk.url,
            'title': chunk.title,
            'content': chunk.content,
            'summary': chunk.summary,
            'metadata': chunk.metadata,
            'embedding': chunk.embedding,
            'chunk_number': chunk.chunk_number
        }).execute()
        
        print(f"Inserted new chunk: {response.data}")
        
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        raise e

async def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python crawl_gdrive_docs.py <folder_id>")
        sys.exit(1)
    
    folder_id = sys.argv[1]
    
    # First clear all data
    await clear_all_data()
    
    # Then process the folder
    await process_folder(folder_id)

if __name__ == "__main__":
    asyncio.run(main())