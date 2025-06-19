import os

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.arxiv import ArxivTools

# Check for API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("🔴 ERROR: Set your GOOGLE_API_KEY environment variable")
    exit(1)

# Create the agent with better error handling
agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    tools=[
        ArxivTools(
            search_arxiv=True,
            read_arxiv_papers=True,
            download_dir="./temp_papers",  # Specify download directory
        )
    ],
    instructions="""
    You are a brilliant research assistant. When given a paper title:
    1. Search for the paper on ArXiv using your tools
    2. If you encounter access issues, try searching with different keywords or variations of the title
    3. Read the paper content when available
    4. Create comprehensive handnotes in this format:
    
    # 🧠 Brilliant Handnotes: {Paper Title}
    
    ## 📜 The Gist
    {One sentence explaining the core idea}
    
    ## 🎯 The Problem
    {What challenge were they solving?}
    
    ## ✨ Key Innovation
    {The main breakthrough or clever idea}
    
    ## 🛠️ How It Works
    {Methodology in plain English}
    
    ## 📊 Results
    {Key findings and what they proved}
    
    ## 🚀 Why It Matters
    {Impact and implications}
    
    ## 🤔 Limitations
    {What are the open questions or limitations?}
    
    If you cannot access the paper due to network issues, provide what analysis you can based on your knowledge of the paper.
    """,
    show_tool_calls=True,
    markdown=True,
    # Add exponential backoff for better error handling
    exponential_backoff=True,
    delay_between_retries=3,  # Wait 3 seconds between retries
    retries=3,
)


# Alternative approach using ArXiv Knowledge Base (more reliable for repeated access)
def create_knowledge_based_agent():
    """Alternative approach using ArXiv Knowledge Base for better reliability"""
    from agno.knowledge.arxiv import ArxivKnowledgeBase
    from agno.vectordb.lancedb import LanceDb

    # Create knowledge base (this approach is more reliable for repeated queries)
    knowledge_base = ArxivKnowledgeBase(
        queries=["Attention is All You Need", "transformer", "attention mechanism"],
        vector_db=LanceDb(table_name="arxiv_papers", uri="./temp_lancedb"),
    )

    # Create agent with knowledge base
    kb_agent = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions="""
        Create brilliant handnotes for papers in the knowledge base using this format:
        
        # 🧠 Brilliant Handnotes: {Paper Title}
        
        ## 📜 The Gist
        {One sentence explaining the core idea}
        
        ## 🎯 The Problem  
        {What challenge were they solving?}
        
        ## ✨ Key Innovation
        {The main breakthrough or clever idea}
        
        ## 🛠️ How It Works
        {Methodology in plain English}
        
        ## 📊 Results
        {Key findings and what they proved}
        
        ## 🚀 Why It Matters
        {Impact and implications}
        
        ## 🤔 Limitations
        {What are the open questions or limitations?}
        """,
        markdown=True,
    )

    return kb_agent, knowledge_base


# Run the agent
if __name__ == "__main__":
    paper_title = "Attention is All You Need"

    print("🤖 Trying ArxivTools approach...")
    print("-" * 60)

    try:
        agent.print_response(f"Analyze the paper: {paper_title}", stream=True)
    except Exception as e:
        print(f"❌ ArxivTools failed: {e}")
        print("\n🔄 Trying Knowledge Base approach...")
        print("-" * 60)

        try:
            kb_agent, knowledge_base = create_knowledge_based_agent()

            # Load the knowledge base (comment out after first run)
            print("📚 Loading knowledge base...")
            knowledge_base.load(recreate=True)

            # Query the knowledge base
            kb_agent.print_response(
                f"Tell me about the '{paper_title}' paper", stream=True
            )

        except Exception as kb_error:
            print(f"❌ Knowledge Base approach also failed: {kb_error}")
            print("\n💡 Fallback: Using general knowledge...")

            # Fallback to general knowledge
            fallback_agent = Agent(model=Gemini(id="gemini-2.0-flash"), markdown=True)

            fallback_agent.print_response(
                f"Based on your knowledge, create handnotes for the paper '{paper_title}' "
                f"using the format I specified earlier. Focus on the transformer architecture "
                f"and attention mechanism.",
                stream=True,
            )
