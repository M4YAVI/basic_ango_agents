from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# 1. --- Define the Agent's Persona and Instructions ---
# The description gives the agent its core identity.
agent_description = dedent(
    """\
    You are a meticulous and unbiased fact-checking agent. Your sole purpose is to
    verify claims by searching for information from multiple, reliable online sources.
    You must remain neutral and present only verified facts."""
)

# The instructions provide a clear, step-by-step process for the agent to follow.
agent_instructions = dedent(
    """\
    1. Acknowledge the user's claim you are about to investigate.
    2. Use the `duckduckgo_search` tool to find information related to the claim.
       Focus on credible sources like established news organizations, scientific journals,
       and official encyclopedias.
    3. Analyze the search results to determine the validity of the claim.
    4. Synthesize the findings into a clear, concise summary.
    5. State your conclusion clearly: "True", "False", "Partially True", or "Unverified".
    6. Provide a brief explanation supporting your conclusion, citing the sources you used."""
)


# 2. --- Build the Agent ---
fact_checker_agent = Agent(
    # Model: We specify the Gemini model to power our agent's reasoning.
    # 'gemini-1.5-flash-latest' is fast and capable for this task.
    model=Gemini(id="gemini-1.5-flash-latest"),
    # Tools: We give the agent the ability to search the web.
    # The DuckDuckGoTools toolkit is simple and requires no extra API keys.
    tools=[DuckDuckGoTools()],
    # Instructions & Description: We pass in the persona and steps we defined above.
    description=agent_description,
    instructions=agent_instructions,
    # Show Tool Calls: This is very useful for seeing the agent's "work".
    # It will print the tool calls it makes during its process.
    show_tool_calls=True,
    # Markdown: This tells the agent to format its final response using markdown
    # for better readability.
    markdown=True,
)


# 3. --- Run the Agent ---
if __name__ == "__main__":
    print("--- Gemini Fact-Checker Agent ---")
    print("Type your claim or question below, or type 'exit' to quit.")

    # Let's start with a classic example claim.
    example_claim = "Is the Great Wall of China visible from space with the naked eye?"
    print(f"\nVerifying claim: '{example_claim}'\n")

    # The agent.print_response() function handles running the agent and
    # streaming the output to the console for a real-time feel.
    fact_checker_agent.print_response(example_claim, stream=True)

    # You can also try other claims:
    # fact_checker_agent.print_response("Does lightning never strike the same place twice?", stream=True)
    # fact_checker_agent.print_response("Did vikings wear horned helmets?", stream=True)
