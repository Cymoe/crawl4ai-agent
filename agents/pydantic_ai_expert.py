from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import numpy as np
import sys

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client, create_client
from typing import List, Dict, Any

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert at both Pydantic AI documentation and data analysis. You have access to:
1. Pydantic AI documentation - including examples, API reference, and other resources
2. Business data files:
   - service_package_data.csv: Information about service packages and their details
   - transaction_data.csv: Transaction records and history
   - Revenue_Metrics_Data (Google Sheet): Monthly revenue metrics including MRR, churn, and customer data

Your job is to help users understand both the Pydantic AI framework and analyze the available data.
When answering questions about data:
1. Pay attention to which file/sheet the user is asking about
2. Focus on the actual numbers and statistics from the specific file
3. Make it clear which data source you're using in your answer

Don't ask the user before taking an action, just do it. Always make sure you look at all relevant sources
before answering unless you have already cached the information.

Always let the user know when you didn't find the answer in the documentation or data - be honest.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def preprocess_query(query: str, ctx: RunContext[PydanticAIDeps]) -> str:
    """Preprocess the query to improve search quality by extracting key concepts."""
    try:
        # Extract key concepts using GPT
        response = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": "Extract 3-5 key technical concepts from this query, separated by commas. Focus on technical terms, framework names, and specific functionality."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        
        concepts = response.choices[0].message.content.split(',')
        
        # Combine original query with key concepts
        expanded_query = f"{query} {' '.join(c.strip() for c in concepts)}"
        return expanded_query
        
    except Exception as e:
        print(f"Error preprocessing query: {e}")
        return query  # Fall back to original query on error

async def search_documents(query: str, ctx: RunContext[PydanticAIDeps], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for relevant documents using Supabase vector search."""
    # Get query embedding
    query_embedding = await get_embedding(query, ctx.deps.openai_client)

    # Use Supabase's built-in vector search
    try:
        # Note: Using match_site_pages RPC function defined in Supabase
        response = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': top_k,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()

        return response.data if response.data else []

    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], query: str) -> str:
    """Retrieve relevant documentation based on the query."""
    # Preprocess the query
    expanded_query = await preprocess_query(query, ctx)

    # Search for relevant documents
    results = await search_documents(expanded_query, ctx)

    if not results:
        return "No relevant documentation found."

    # Format results
    formatted_results = []
    for doc in results:
        formatted_results.append(f"From {doc['url']}:\n{doc['content']}\n")

    return "\n".join(formatted_results)

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

async def test_rag(query: str):
    """Test the RAG system with a query."""
    # Initialize clients
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    
    # Create dependencies
    deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)
    
    # Create context with model and empty usage/prompt
    ctx = RunContext(
        deps=deps,
        model=model,
        usage={},
        prompt=""
    )
    
    # Get relevant documentation
    docs = await retrieve_relevant_documentation(ctx, query)
    print("\nRelevant documentation found:")
    print("-" * 50)
    print(docs)
    print("-" * 50)

if __name__ == "__main__":
    import sys
    
    # Use command line argument if provided, otherwise use default query
    query = sys.argv[1] if len(sys.argv) > 1 else "How do I create a new agent in Pydantic AI?"
    print(f"\nTesting RAG with query: {query}")
    print("=" * 50)
    asyncio.run(test_rag(query))