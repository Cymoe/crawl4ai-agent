import os
import json
import asyncio
import tempfile
import traceback
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
import logging
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
    insert_chunk,
    get_embedding  # Import the get_embedding function
)

# Load environment variables
load_dotenv()

# Initialize OpenAI and Supabase clients (reusing from existing code)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Initialize logger
logger = logging.getLogger(__name__)

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

async def process_gdrive_spreadsheet(service, file_id: str, name: str) -> List[ProcessedChunk]:
    """Process a Google Drive spreadsheet."""
    chunks = []
    try:
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
        chunk = await process_dataframe(df, name)
        chunks.append(chunk)
        
        print(f"Processed {len(chunks)} sheets")
        
    except Exception as e:
        print(f"Error processing spreadsheet: {e}")
        return []
        
    return chunks

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
    
    # Generate embedding
    embedding = await get_embedding(content)
    
    return ProcessedChunk(
        title=file_name,
        url=f"gdrive://{file_name}",
        content=content,
        metadata=metadata,
        summary=summary,
        chunk_number=0,
        embedding=embedding
    )

async def process_file(service, file):
    """Process a file from Google Drive."""
    try:
        file_id = file['id']
        name = file['name']
        mime_type = file.get('mimeType', 'unknown')
        
        logger.info(f"Processing file: {name} (ID: {file_id})")
        logger.info(f"MIME type: {mime_type}")
        
        chunks = []
        
        # Handle Google Sheets
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            logger.info(f"Processing Google Sheet: {name}")
            chunks = await process_gdrive_spreadsheet(service, file_id, name)
            # return
        
        # Handle CSV files
        elif mime_type == 'text/csv' or name.lower().endswith('.csv'):
            logger.info(f"Processing CSV file: {name}")
            
            # Download the file
            request = service.files().get_media(fileId=file_id)
            content = request.execute().decode('utf-8')
            
            # Parse CSV data
            try:
                reader = csv.reader(content.splitlines())
                rows = list(reader)
                
                if not rows:
                    logger.warning(f"CSV file {name} is empty")
                    return
                
                headers = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []
                
                # Determine file type from name
                file_type = "unknown"
                if "customer" in name.lower():
                    file_type = "customers"
                elif "transaction" in name.lower():
                    file_type = "transactions"
                elif "geographic" in name.lower() or "geo" in name.lower():
                    file_type = "geographical"
                elif "operation" in name.lower() or "ops" in name.lower():
                    file_type = "operations"
                elif "service" in name.lower() or "package" in name.lower():
                    file_type = "service_packages"
                elif "revenue" in name.lower() or "metric" in name.lower():
                    file_type = "revenue_metrics"
                elif "clean" in name.lower() or "company" in name.lower():
                    file_type = "cleaning_company"
                
                # Create summary
                summary = f"Data from {name} with {len(data_rows)} rows and {len(headers)} columns"
                
                # Create content
                content = f"File: {name}\nType: {file_type}\n\nHeaders: {', '.join(headers)}\n\nSample Data:\n"
                for i, row in enumerate(data_rows[:5]):  # Include up to 5 rows as sample
                    content += f"Row {i+1}: {', '.join(row)}\n"
                
                # Add full data representation
                content += "\nFull Data:\n"
                for i, row in enumerate(data_rows):
                    row_data = []
                    for j, cell in enumerate(row):
                        if j < len(headers):
                            row_data.append(f"{headers[j]}: {cell}")
                        else:
                            row_data.append(f"Column {j+1}: {cell}")
                    content += f"Row {i+1}: {', '.join(row_data)}\n"
                
                # Create chunk
                chunk = ProcessedChunk(
                    url=f"gdrive://{name}",
                    title=name,
                    summary=summary,
                    content=content,
                    embedding=await get_embedding(content),
                    chunk_number=1,
                    metadata={
                        "type": file_type,
                        "source": "gdrive",
                        "file_name": name,
                        "file_type": "spreadsheet"
                    }
                )
                chunks = [chunk]
                
            except Exception as e:
                logger.error(f"Error processing CSV file {name}: {e}")
                logger.error(traceback.format_exc())
                # Create a simple chunk with error information
                chunk = ProcessedChunk(
                    url=f"gdrive://{name}",
                    title=name,
                    summary=f"Error processing CSV file: {name}",
                    content=f"Error processing CSV file: {name}\nError: {str(e)}",
                    chunk_number=1,
                    metadata={
                        "type": "error",
                        "source": "gdrive",
                        "file_name": name,
                        "file_type": "spreadsheet",
                        "error": str(e)
                    }
                )
                chunks = [chunk]
        
        # Handle text files
        elif mime_type == 'text/plain' or name.lower().endswith('.txt'):
            logger.info(f"Processing text file: {name}")
            try:
                request = service.files().get_media(fileId=file_id)
                content = request.execute().decode('utf-8')
                
                # Determine file type from name
                file_type = "text"
                if "clean" in name.lower() or "company" in name.lower():
                    file_type = "cleaning_company"
                
                # Process as raw text
                chunk = ProcessedChunk(
                    url=f"gdrive://{name}",
                    title=name,
                    summary=f"Text file: {name}",
                    content=content,
                    embedding=await get_embedding(content),
                    chunk_number=1,
                    metadata={
                        "source": "gdrive",
                        "type": file_type,
                        "file_name": name,
                        "file_type": "text",
                        "file_id": file_id,
                        "mime_type": mime_type
                    }
                )
                chunks = [chunk]
            except Exception as e:
                logger.error(f"Error processing text file {name}: {e}")
                logger.error(traceback.format_exc())
                chunk = ProcessedChunk(
                    url=f"gdrive://{name}",
                    title=name,
                    summary=f"Error processing text file: {name}",
                    content=f"Error processing text file: {name}\nError: {str(e)}",
                    embedding=await get_embedding(f"Error processing text file: {name}\nError: {str(e)}"),
                    chunk_number=1,
                    metadata={
                        "type": "error",
                        "source": "gdrive",
                        "file_name": name,
                        "file_type": "text",
                        "error": str(e)
                    }
                )
                chunks = [chunk]
        
        # Handle Numbers files (Apple's spreadsheet format)
        elif mime_type == 'application/vnd.apple.numbers' or name.lower().endswith('.numbers'):
            logger.info(f"Processing Numbers file: {name}")
            try:
                request = service.files().get_media(fileId=file_id)
                content = request.execute()
                
                # Save temporarily and process
                with tempfile.NamedTemporaryFile(suffix='.numbers') as temp_file:
                    temp_file.write(content)
                    temp_file.flush()
                    
                    # Convert Numbers content to text representation
                    text_content = f"Numbers spreadsheet: {name}\n\n"
                    text_content += "Note: This is a Numbers spreadsheet file. Please convert to CSV or Google Sheets for full data processing."
                    
                    # Determine file type from name
                    file_type = "spreadsheet"
                    if "test" in name.lower() or "metric" in name.lower():
                        file_type = "test_metrics"
                    
                    chunk = ProcessedChunk(
                        url=f"gdrive://{name}",
                        title=name,
                        summary=f"Numbers spreadsheet: {name}",
                        content=text_content,
                        embedding=await get_embedding(text_content),
                        chunk_number=1,
                        metadata={
                            "source": "gdrive",
                            "type": file_type,
                            "file_name": name,
                            "file_type": "spreadsheet",
                            "file_id": file_id,
                            "mime_type": mime_type
                        }
                    )
                    chunks = [chunk]
            except Exception as e:
                logger.error(f"Error processing Numbers file {name}: {e}")
                logger.error(traceback.format_exc())
                chunk = ProcessedChunk(
                    url=f"gdrive://{name}",
                    title=name,
                    summary=f"Error processing Numbers file: {name}",
                    content=f"Error processing Numbers file: {name}\nError: {str(e)}",
                    embedding=await get_embedding(f"Error processing Numbers file: {name}\nError: {str(e)}"),
                    chunk_number=1,
                    metadata={
                        "type": "error",
                        "source": "gdrive",
                        "file_name": name,
                        "file_type": "spreadsheet",
                        "error": str(e)
                    }
                )
                chunks = [chunk]
        
        else:
            logger.warning(f"Unsupported file type: {mime_type} for file {name}")
            # Create a simple chunk with file information
            chunk = ProcessedChunk(
                url=f"gdrive://{name}",
                title=name,
                summary=f"Unsupported file type: {mime_type}",
                content=f"File {name} has unsupported type: {mime_type}",
                embedding=await get_embedding(f"File {name} has unsupported type: {mime_type}"),
                chunk_number=1,
                metadata={
                    "type": "unsupported",
                    "source": "gdrive",
                    "file_name": name,
                    "file_type": "unsupported",
                    "mime_type": mime_type
                }
            )
            chunks = [chunk]
            
        for chunk in chunks:
            logger.info(f"Inserting chunk {chunk.chunk_number} for {chunk.url}")
            
            # Insert into database
            await insert_chunk(chunk)
        
        # Verify the file was stored
        logger.info("Verifying database storage...")
        await list_gdrive_files()
            
    except Exception as e:
        logger.error(f"Error processing file {file.get('name', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to allow caller to handle

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
            "application/vnd.ms-excel",
            "text/plain",
            "application/vnd.apple.numbers"
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
        logger.error(f"Error getting embedding: {str(e)}")
        return [0.0] * 1536  # Return zero vector on failure

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

async def delete_file_from_database(file_id: str, file_name: str = None):
    """Delete a file from the database when it's removed from Google Drive."""
    try:
        print(f"Deleting file with ID {file_id} from database")
        
        # Get all gdrive files
        response = supabase.table('site_pages').select('*').execute()
        all_data = response.data
        
        # Filter to only gdrive files
        gdrive_data = [
            item for item in all_data 
            if item.get('metadata', {}).get('source') == 'gdrive'
        ]
        
        deleted = False
        
        # First try to find by file_id in metadata
        for item in gdrive_data:
            metadata = item.get('metadata', {})
            if metadata.get('file_id') == file_id:
                # Delete the record
                delete_response = supabase.table('site_pages').delete().eq('id', item['id']).execute()
                print(f"Deleted record with ID {item['id']} by file_id")
                deleted = True
        
        # If not found and we have a file name, try by file name
        if not deleted and file_name:
            for item in gdrive_data:
                metadata = item.get('metadata', {})
                if metadata.get('file_name') == file_name:
                    # Delete the record
                    delete_response = supabase.table('site_pages').delete().eq('id', item['id']).execute()
                    print(f"Deleted record with ID {item['id']} by file_name")
                    deleted = True
        
        # Last resort: try by URL
        if not deleted and file_name:
            url_pattern = f"gdrive://{file_name}"
            for item in gdrive_data:
                if item.get('url') == url_pattern:
                    # Delete the record
                    delete_response = supabase.table('site_pages').delete().eq('id', item['id']).execute()
                    print(f"Deleted record with ID {item['id']} by URL")
                    deleted = True
        
        # If we still haven't found it, try the title
        if not deleted and file_name:
            for item in gdrive_data:
                if item.get('title') == file_name:
                    # Delete the record
                    delete_response = supabase.table('site_pages').delete().eq('id', item['id']).execute()
                    print(f"Deleted record with ID {item['id']} by title")
                    deleted = True
        
        if not deleted:
            print(f"No records found for file ID {file_id} or name {file_name}")
            
    except Exception as e:
        print(f"Error deleting file from database: {e}")
        import traceback
        print(f"Stack trace: {traceback.format_exc()}")

async def list_gdrive_files():
    """List all Google Drive files stored in the database."""
    print("\n=== Files in Database ===")
    try:
        response = supabase.table('site_pages').select('*').execute()
        all_data = response.data
        
        # Filter to only gdrive files
        gdrive_data = [
            item for item in all_data 
            if item.get('metadata', {}).get('source') == 'gdrive'
        ]
        
        print(f"\nFound {len(gdrive_data)} Google Drive files:")
        for item in gdrive_data:
            print(f"\nFile: {item.get('title')}")
            print(f"Type: {item.get('metadata', {}).get('type', 'unknown')}")
            print(f"URL: {item.get('url')}")
            print(f"Summary: {item.get('summary')}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing files: {e}")
        print(f"Stack trace: {traceback.format_exc()}")

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