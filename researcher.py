import os
from textwrap import dedent
from typing import List

from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.newspaper4k import Newspaper4kTools
from pydantic import BaseModel, Field
from rich.pretty import pprint

# Check if API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Error: GOOGLE_API_KEY environment variable is not set!")
    print("Please set your Google API key:")
    print("export GOOGLE_API_KEY=your_api_key_here")
    print("\nGet your API key from: https://ai.google.dev/gemini-api/docs/api-key")
    exit(1)


# 1. --- Define the "Cool Notes" Structure ---
# This Pydantic model is the blueprint for our agent's output.
class ResearchPaperNotes(BaseModel):
    """Structured, insightful notes from an AI research paper."""

    title: str = Field(..., description="The official title of the research paper.")

    tldr: str = Field(
        ...,
        description="A single, impactful sentence summarizing the entire paper. The 'Too Long; Didn't Read' version.",
    )

    key_contributions: List[str] = Field(
        ...,
        description="A bulleted list of the 3 most important contributions or findings of the paper.",
    )

    methodology_simplified: str = Field(
        ...,
        description="An easy-to-understand explanation of the paper's core methodology, as if explaining it to a smart colleague from a different field.",
    )

    impact_and_significance: str = Field(
        ...,
        description="A paragraph explaining why this paper is important and what its potential impact on the field of AI could be.",
    )

    original_paper_url: str = Field(
        ..., description="The original URL of the research paper that was analyzed."
    )


# 2. --- Build the Agent ---
research_agent = Agent(
    # Model: Gemini 1.5 Pro is excellent for digesting and synthesizing dense text.
    model=Gemini(id="gemini-2.0-flash"),
    # Tools: Newspaper4kTools can read and parse the content from a URL.
    tools=[Newspaper4kTools()],
    # Description: A professional and expert persona.
    description=dedent(
        """\
        You are Dr. Axiom, a world-renowned AI research analyst. Your unique talent
        is distilling complex, dense academic papers into clear, insightful, and
        actionable notes for busy engineers and researchers."""
    ),
    # Instructions: A clear step-by-step process.
    instructions=dedent(
        """\
        1. The user will provide a URL to an AI research paper.
        2. Use the `read_article` tool to ingest the paper's content.
        3. Carefully analyze the text to understand its core concepts, methodology, and findings.
        4. Populate ALL fields of the `ResearchPaperNotes` structure with your analysis.
        5. Your final output must ONLY be the structured data, with no additional commentary."""
    ),
    # Response Model: We enforce our "cool notes" structure.
    response_model=ResearchPaperNotes,
    # Show Tool Calls: Useful for seeing the agent in action.
    show_tool_calls=True,
)


# 3. --- Run the Agent ---
if __name__ == "__main__":
    # Let's use the foundational "Attention Is All You Need" paper as our example.
    paper_url = "https://arxiv.org/abs/1706.03762"

    print("--- 🔬 Gemini Research Paper Distiller ---")
    print(f"Analyzing paper: {paper_url}\n")
    print(
        "🤖 Dr. Axiom is reading and analyzing the paper... this may take a moment.\n"
    )

    try:
        # The prompt for the agent is simple: just the URL.
        # We pass the URL again in the context so the agent can easily populate the 'original_paper_url' field.
        response: RunResponse = research_agent.run(
            paper_url,
            # We can pass context to the agent, which it can use in its response.
            context={"original_paper_url": paper_url},
        )

        notes = response.content

        # Check if the output is the Pydantic object we expect
        if isinstance(notes, ResearchPaperNotes):
            print("--- ✨ Here are your Cool Notes from Dr. Axiom ✨ ---\n")
            print(f"📄 Title: {notes.title}\n")
            print(f"🧠 TL;DR: {notes.tldr}\n")
            print("🎯 Key Contributions:")
            for contribution in notes.key_contributions:
                print(f"   - {contribution}")
            print("\n⚙️ Methodology, Simplified:")
            print(f"   {notes.methodology_simplified}\n")
            print("🚀 Impact and Significance:")
            print(f"   {notes.impact_and_significance}\n")
            print(f"🔗 Original Paper: {notes.original_paper_url}")
        else:
            print("Sorry, there was an error processing the paper.")
            pprint(notes)

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check your API key and internet connection.")
