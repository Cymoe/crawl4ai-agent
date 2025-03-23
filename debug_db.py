import os
from supabase import create_client, Client
import json

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def main():
    print("=== Checking Database Content ===")
    
    # Get all documents
    response = supabase.table('site_pages').select('*').execute()
    documents = response.data
    
    print(f"\nFound {len(documents)} total documents")
    
    for doc in documents:
        print("\n--- Document ---")
        print(f"Source: {doc.get('source')}")
        print(f"Title: {doc.get('title')}")
        print(f"URL: {doc.get('url')}")
        print(f"Metadata: {json.dumps(doc.get('metadata'), indent=2)}")
        print("\nContent Preview:")
        content = doc.get('content', '')
        print(content[:500] + "..." if len(content) > 500 else content)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
