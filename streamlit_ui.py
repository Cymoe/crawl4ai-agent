import streamlit as st
from agents.pydantic_ai_expert import pydantic_ai_expert
import asyncio
import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
from supabase import create_client, Client
import traceback
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import uvicorn
from starlette.responses import JSONResponse

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Create a FastAPI app
api = FastAPI()

# Add API endpoints
@api.get("/api/check_database")
async def check_database():
    """API endpoint to check the database."""
    try:
        response = supabase.table('site_pages').select('*').execute()
        all_data = response.data
        
        # Filter to only gdrive files
        gdrive_data = [
            item for item in all_data 
            if item.get('metadata', {}).get('source') == 'gdrive'
        ]
        
        return gdrive_data
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()}
        )

async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI API."""
    response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

async def search_data(query: str) -> List[Dict[str, Any]]:
    """Search for relevant data in Supabase using vector similarity."""
    try:
        # Generate embedding for the query
        embedding = await get_embedding(query)
        
        print("\n=== Searching Database with Vector Similarity ===")
        print(f"Query: {query}")
        
        # Use Supabase's vector similarity search with the embedding
        # This will find documents that are semantically similar to the query
        try:
            response = supabase.rpc(
                'match_site_pages',  
                {
                    'query_embedding': embedding,
                    'match_threshold': 0.5,  
                    'match_count': 10        
                }
            ).execute()
            
            # Check if we got any results
            if response.data and len(response.data) > 0:
                results = response.data
                print(f"Found {len(results)} documents through vector similarity")
                
                # Log the results for debugging
                for result in results:
                    print(f"\n--- Document ---")
                    print(f"Title: {result.get('title')}")
                    print(f"Type: {result.get('metadata', {}).get('type')}")
                    print(f"Content Preview: {result.get('content')[:100]}...")
                
                return results
            else:
                print("No vector matches found, falling back to keyword search")
                # Continue to fallback method
        except Exception as e:
            print(f"Vector search error: {e}")
            print("Falling back to keyword search")
            # Continue to fallback method
        
        # Fallback: Get all documents and filter
        response = supabase.table('site_pages').select('*').execute()
        all_data = response.data
        
        # Filter to only gdrive files
        gdrive_data = [
            item for item in all_data 
            if item.get('metadata', {}).get('source') == 'gdrive'
        ]
        
        print(f"Found {len(gdrive_data)} gdrive documents")
        
        # If asking about metrics, prioritize revenue metrics data
        metrics = ['mrr', 'revenue', 'churn', 'customers', 'cac', 'ltv']
        if any(metric in query.lower() for metric in metrics):
            results = [
                item for item in gdrive_data
                if item.get('metadata', {}).get('type') == 'revenue_metrics'
            ]
            if not results:  # If no revenue metrics found, use all gdrive data
                results = gdrive_data
        else:
            # Use simple keyword matching as fallback
            query_terms = query.lower().split()
            results = []
            
            for item in gdrive_data:
                content = item.get('content', '').lower()
                title = item.get('title', '').lower()
                # Check if any query term appears in content or title
                if any(term in content or term in title for term in query_terms):
                    results.append(item)
            
            # If still no results, return all gdrive data
            if not results:
                results = gdrive_data
        
        # Log the results for debugging
        print(f"\nReturning {len(results)} relevant documents")
        for result in results:
            print(f"\n--- Document ---")
            print(f"Title: {result.get('title')}")
            print(f"Type: {result.get('metadata', {}).get('type')}")
            print(f"Content Preview: {result.get('content')[:100]}...")
        
        return results
    except Exception as e:
        print(f"Error searching data: {e}")
        traceback.print_exc()
        return []

def extract_relevant_content(content: str, query: str) -> str:
    """Extract only the relevant parts of the content based on the query."""
    # Split content into sections
    sections = content.split('\n# ')
    
    # If asking about specific metrics, only return the raw data section
    metrics = ['mrr', 'revenue', 'churn', 'customers', 'cac', 'ltv']
    if any(metric in query.lower() for metric in metrics):
        for section in sections:
            if section.startswith('Raw Data'):
                return '# ' + section
    
    # Otherwise return a truncated version of the full content
    return content[:4000]  # Limit to ~1000 tokens

def process_raw_data(content: str) -> pd.DataFrame:
    """Convert raw data text back into a DataFrame for calculations."""
    if '# Raw Data' not in content:
        return None
        
    # Extract the raw data section
    data_section = content.split('# Raw Data')[1].split('#')[0]
    
    # Parse the records
    records = []
    current_record = {}
    
    for line in data_section.split('\n'):
        line = line.strip()
        if line.startswith('Record:'):
            if current_record:
                records.append(current_record)
            current_record = {}
        elif line.startswith('- '):
            try:
                key, value = line[2:].split(': ', 1)
                # Clean up the value - remove $ and % and convert to float if possible
                value = value.replace('$', '').replace(',', '').replace('%', '')
                try:
                    value = float(value)
                except:
                    pass
                current_record[key] = value
            except:
                continue
                
    if current_record:
        records.append(current_record)
        
    return pd.DataFrame(records)

def calculate_metrics(df: pd.DataFrame, query_type: str = None) -> str:
    """Calculate metrics for a specific time period."""
    metrics = []
    
    if query_type == 'mrr_growth':
        if 'MRR' in df.columns:
            mrr_values = df['MRR'].dropna()
            if len(mrr_values) > 1:
                total_growth = ((mrr_values.iloc[-1] / mrr_values.iloc[0]) - 1) * 100
                metrics.append("MRR Growth Analysis:")
                metrics.append(f"- Starting MRR: ${mrr_values.iloc[0]:,.2f}")
                metrics.append(f"- Current MRR: ${mrr_values.iloc[-1]:,.2f}")
                metrics.append(f"- Growth Rate: {total_growth:.1f}%")
                
                # Add month-by-month growth
                metrics.append("\nMonth-by-Month MRR:")
                for i, mrr in enumerate(mrr_values):
                    metrics.append(f"- {df['Month'].iloc[i]}: ${mrr:,.2f}")
    
    elif query_type == 'churn':
        if 'Churn Rate' in df.columns:
            churn_values = df['Churn Rate'].dropna()
            avg_churn = churn_values.mean()
            metrics.append("Churn Rate Analysis:")
            metrics.append(f"- Average Churn Rate: {avg_churn:.2f}%")
            
            # Add month-by-month churn
            metrics.append("\nMonth-by-Month Churn Rates:")
            for i, churn in enumerate(churn_values):
                metrics.append(f"- {df['Month'].iloc[i]}: {churn:.2f}%")
            
            metrics.append(f"\nHighest Churn: {churn_values.max():.2f}%")
            metrics.append(f"Lowest Churn: {churn_values.min():.2f}%")
    
    return "\n".join(metrics)

async def get_assistant_response(query: str, context: List[Dict[str, Any]]) -> str:
    """Get assistant response using OpenAI."""
    system_prompt = """You are an AI assistant that helps users understand their business data. 
    You will be given context from various data sources, and your task is to answer the user's question based on this context.
    
    If the information is not present in the context, say that you don't have enough information to answer.
    Do not make up information that is not in the context.
    
    When analyzing data, provide insights that would be valuable for business decision-making.
    If appropriate, suggest follow-up questions that might help the user gain deeper insights.
    
    Remember to maintain a professional but friendly tone."""
    
    # Format context for the prompt, being selective about content
    formatted_context = ""
    dataframes = {}  # Store processed dataframes for calculations
    
    # Limit the number of context items to prevent token overflow
    max_context_items = 5
    context = context[:max_context_items]
    
    for item in context:
        content = item.get('content', '')
        
        # Process the raw data into a DataFrame if possible
        df = process_raw_data(content)
        if df is not None:
            dataframes[item.get('title', 'unknown')] = df
        
        # Extract only relevant parts of the content and limit its size
        relevant_content = extract_relevant_content(content, query)
        # Limit content size to prevent token overflow
        if len(relevant_content) > 1000:
            relevant_content = relevant_content[:1000] + "... [content truncated]"
            
        formatted_context += f"\nSource: {item.get('metadata', {}).get('file_name', 'Unknown')}\n"
        formatted_context += f"Content:\n{relevant_content}\n"
        formatted_context += "---\n"
    
    # Check for specific calculation requests
    query_lower = query.lower()
    calculation_type = None
    
    if 'mrr' in query_lower and 'growth' in query_lower:
        calculation_type = 'mrr_growth'
    elif 'churn' in query_lower and any(word in query_lower for word in ['average', 'rate', 'trend']):
        calculation_type = 'churn'
    
    if calculation_type:
        # Find the revenue metrics data
        for title, df in dataframes.items():
            if 'Revenue_Metrics_Data' in title:
                formatted_context += "\nCalculated Metrics:\n"
                formatted_context += calculate_metrics(df, calculation_type)
                break
    
    print("\n=== Context Being Sent to GPT ===")
    print(formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the relevant context:\n\n{formatted_context}\n\nQuestion: {query}"}
    ]
    
    # Get model from environment variables or use a default
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return f"I encountered an error processing your request. Please try again with a simpler query or less data. Error details: {str(e)[:100]}..."

def run_api():
    """Run the FastAPI app."""
    app = WSGIMiddleware(api)
    uvicorn.run(app, host="0.0.0.0", port=8000)

async def main():
    st.title("AMS Clean Assistant")
    st.write("Ask me anything about AMS Clean's data and documents.")
    
    # Add sidebar to display available files
    with st.sidebar:
        st.header("Available Files")
        st.write("The following files are available for querying:")
        
        try:
            # Get all Google Drive files from the database
            response = supabase.table('site_pages').select('*').execute()
            all_data = response.data
            
            # Filter to only gdrive files
            gdrive_data = [
                item for item in all_data 
                if item.get('metadata', {}).get('source') == 'gdrive'
            ]
            
            # Display files in the sidebar
            for item in gdrive_data:
                file_name = item.get('metadata', {}).get('file_name', item.get('title', 'Unknown'))
                file_type = item.get('metadata', {}).get('type', 'unknown')
                st.write(f" **{file_name}** ({file_type})")
                
            st.write("---")
            st.write("Last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("How can I help you?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing data..."):
                # Search for relevant data
                results = await search_data(prompt)
                
                # Get assistant response
                response = await get_assistant_response(prompt, results)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Start the API server in a separate thread
    import threading
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()

if __name__ == "__main__":
    asyncio.run(main())
