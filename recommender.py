from textwrap import dedent
from typing import List

from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from rich.pretty import pprint


# 1. --- Define the Structured Output (our Pydantic Model) ---
# This tells the agent EXACTLY what format we expect the data in.
class Recommendation(BaseModel):
    """A single movie or book recommendation."""

    title: str = Field(..., description="The title of the movie or book.")
    year: int = Field(..., description="The year the item was released.")
    summary: str = Field(..., description="A one-sentence summary of the plot.")
    reason: str = Field(
        ...,
        description="A short, compelling reason why the user would like this, based on their original prompt.",
    )


class RecommendationList(BaseModel):
    """A list of three recommendations."""

    recommendations: List[Recommendation] = Field(
        ..., description="A list of exactly three recommendations."
    )


# 2. --- Build the Agent ---
recommender_agent = Agent(
    # Model: We'll use Gemini for its excellent knowledge base.
    model=Gemini(id="gemini-1.5-flash-latest"),
    # Description: This gives the agent its personality.
    description=dedent(
        """\
        You are 'Cine-Bot 3000', a super-enthusiastic movie and book aficionado.
        Your passion is finding the perfect recommendation for any user."""
    ),
    # Instructions: Guide the agent's thinking process.
    instructions="Based on the user's favorite movie or book, generate three similar recommendations that they will absolutely love.",
    # Response Model: This is the magic! We tell the agent its final output MUST
    # conform to our `RecommendationList` Pydantic model.
    response_model=RecommendationList,
)


# 3. --- Run the Agent ---
if __name__ == "__main__":
    prompt = "I just watched 'Dune' and loved the epic world-building and political intrigue. What should I watch or read next?"
    print(f"--- Gemini Recommender Agent ---")
    print(f"User Prompt: {prompt}\n")
    print("🤖 Cine-Bot is thinking...\n")

    # We use agent.run() to get the structured data object back.
    response: RunResponse = recommender_agent.run(prompt)

    # The 'response.content' will be our Pydantic object!
    recommendations_object = response.content

    print("--- Raw Pydantic Object ---")
    pprint(recommendations_object)
    print("\n" + "=" * 30 + "\n")

    # Now we can process the structured data in a reliable way.
    if isinstance(recommendations_object, RecommendationList):
        print("--- Here are my recommendations for you! ---\n")
        for rec in recommendations_object.recommendations:
            print(f"🎬 Title: {rec.title} ({rec.year})")
            print(f"   📜 Summary: {rec.summary}")
            print(f"   🤔 Why you'll like it: {rec.reason}\n")
    else:
        print("Sorry, I couldn't generate recommendations in the correct format.")
