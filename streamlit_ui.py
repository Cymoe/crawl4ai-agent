import streamlit as st
from agents.pydantic_ai_expert import pydantic_ai_expert
import asyncio
import os
from typing import List, Dict, Any
from openai import AsyncOpenAI
from supabase import create_client, Client
import traceback
import pandas as pd

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI API."""
    response = await openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

async def search_data(query: str) -> List[Dict[str, Any]]:
    """Search for relevant data in Supabase."""
    embedding = await get_embedding(query)
    
    try:
        # First, let's check what data exists
        print("\n=== Checking Database Content ===")
        response = supabase.table('site_pages').select('*').execute()
        all_data = response.data
        print(f"Total records found: {len(all_data)}")
        
        # Filter to only gdrive files
        gdrive_data = [
            item for item in all_data 
            if item.get('metadata', {}).get('source') == 'gdrive'
        ]
        
        print(f"\nFound {len(gdrive_data)} gdrive documents")
        
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
            results = gdrive_data
        
        print(f"\nReturning {len(results)} relevant documents")
        for result in results:
            print(f"\n--- Document ---")
            print(f"Title: {result.get('title')}")
            print(f"Type: {result.get('metadata', {}).get('type')}")
            print(f"Content Preview: {result.get('content')[:200]}...")
        
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
    system_prompt = """You are AMS Clean Assistant, a data analyst for AMS Clean. 
    Your role is to help answer questions about company data and perform calculations when needed.
    
    When analyzing data:
    1. Focus on providing clear, direct answers with specific numbers
    2. Format monetary values with $ and commas (e.g. $1,234.56)
    3. Format percentages with % sign (e.g. 12.3%)
    4. If asked to perform calculations:
       - Show the formula being used (if applicable)
       - List out the values being calculated
       - Show the step-by-step calculation
       - Present the final result clearly
    5. If data is missing or unclear, say so directly
    
    Remember to maintain a professional but friendly tone."""
    
    # Format context for the prompt, being selective about content
    formatted_context = ""
    dataframes = {}  # Store processed dataframes for calculations
    
    for item in context:
        content = item.get('content', '')
        
        # Process the raw data into a DataFrame if possible
        df = process_raw_data(content)
        if df is not None:
            dataframes[item.get('title', 'unknown')] = df
        
        # Extract only relevant parts of the content
        relevant_content = extract_relevant_content(content, query)
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
    
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )
    
    return response.choices[0].message.content

async def main():
    st.title("AMS Clean Assistant")
    st.write("Ask me anything about AMS Clean's data and documents.")
    
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

if __name__ == "__main__":
    asyncio.run(main())
