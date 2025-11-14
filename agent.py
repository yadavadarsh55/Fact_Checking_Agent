from langchain.tools import tool
from langchain_community.document_loaders import WikipediaLoader
from rake_nltk import Rake
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain_core.runnables import Runnable
from langchain_core.agents import AgentAction, AgentFinish
from typing import Union, List, Any, Dict

load_dotenv()

AgentRunnable = Runnable[Union[Dict[str, Any], List[Any]], Union[AgentAction, AgentFinish]]

@tool
def infromation_retrieving_tool(claim: str) -> list:
    """
    A tool to retirieve the infromation about the particular given statement.
    
    Args:
        claim (str): The claim or information to be fetched.
        
    Returns:
        list: The list in the format of langchain document having the infromation about the query.
    """
    
    r = Rake()
    r.extract_keywords_from_text(claim)
    topic = r.get_ranked_phrases()[:3]

    if topic == "":
        raise ValueError("No topic found in the claim.")
    
    loader = WikipediaLoader(topic, load_max_docs=1)
    documents = loader.load()
    return documents

SYSTEM_PROMPT = """
    You are a fact-checking agent. Your task is to verify the claims made by users using reliable sources such as Wikipedia. When a user provides a claim, follow these steps:
    1. Use the 'getTopicTool' to extract the main topic from the claim.
    2. Use the 'infromation_retrieving_tool' to fetch relevant information about the topic from Wikipedia.
    3. Compare the claim with the retrieved information to determine its accuracy.
    4. Provide a clear and concise response indicating whether the claim is true, false, or unverifiable based on the information found.
    You have access to the tool:
    infromation_retrieving_tool: use this to get the relevant information about the claim provided by the user.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=1000
)

@dataclass
class ResonseFormat:
    '''Response schema for the fact-checking agent'''
    fact_check_response: str

def initialize_agent() -> AgentRunnable:
    """
    Function to initialize and return the fact-checking agent.

    Returns:
        AgentRunnable: The initialized fact-checking agent.
    """
    agent = create_agent(
            model=llm,
            tools=[infromation_retrieving_tool],
            system_prompt=SYSTEM_PROMPT,
            response_format=ResonseFormat,
        )
    return agent

def get_Claim():
    """
    Function to get the claim from the user.

    Returns:
        str: The claim provided by the user.
    """
    try:
        claim = input("Enter the claim to be fact-checked: ").strip()
        if not claim:
            raise ValueError("Claim cannot be empty")
        return claim
    except Exception as e:
        print(f"Error getting claim: {str(e)}")
        return None

def fact_check_claim(agent: AgentRunnable, claim: str) -> str:
    """
    Function to fact-check a claim using the fact-checking agent.

    Args:
        agent (AgentRunnable): The fact-checking agent.
        claim (str): The claim or information provided by the user.

    Returns:
        str: The fact-checked response indicating the accuracy of the claim.
    """
    if not claim:
        return "No claim provided to fact check."
    
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": claim}]}
        )
        return response['structured_response'].fact_check_response
    except Exception as e:
        return f"Error during fact checking: {str(e)}"
    

__all__ = [
    'initialize_agent', 
    'fact_check_claim' 
]