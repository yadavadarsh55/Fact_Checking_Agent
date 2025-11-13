import streamlit as st
import os

# Import necessary components from the agent module
from agent import (
    initialize_agent,
    fact_check_claim 
)

# --- Configuration ---

st.set_page_config(
    page_title="Fact Checker Agent",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Agent Initialization (Cached for performance) ---

@st.cache_resource
def get_agent():
    """Initializes and returns the fact-checking agent using caching."""
    # Ensure the required environment variable is set before creating the agent
    if not os.getenv("GOOGLE_API_KEY"):
        return None

    try:
        # Call the centralized function from agent.py
        agent = initialize_agent() 
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

def fact_check_claim_streamlit(claim: str):
    """Invokes the cached agent and returns the structured response or error."""
    agent = get_agent()
    
    if agent is None:
        return "Agent failed to initialize. Please ensure GEMINI_API_KEY is set."
    
    # Pass the agent object to the core fact_check_claim function
    result = fact_check_claim(agent, claim)
    return result

# --- Streamlit UI ---

st.title("💡 Fact Checker Agent")
st.markdown("Enter a claim below to have the agent verify its accuracy using up-to-date information from Wikipedia.")

# Check for API key and initialize agent
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("🚨 GEMINI_API_KEY environment variable not set. Please set it to run the agent.")
else:
    with st.spinner("Initializing fact-checking agent..."):
        agent_status = get_agent()
    
    if agent_status:
        st.success("Agent ready! Submit your claim.")
    else:
        st.error("Agent failed to initialize. Check console for details.")

# User input for the claim
user_claim = st.text_input(
    "Enter the claim to verify:",
    placeholder="Example: Earth is flat.",
    key="claim_input"
)

# Button to trigger the fact-check
if st.button("Verify Claim", type="primary") and user_claim:
    st.subheader("Verification Result")
    
    with st.spinner(f"Verifying claim: '{user_claim}'..."):
        # Invoke the fact-checking process
        result = fact_check_claim_streamlit(user_claim)
        
        # Simple heuristic to determine true/false for styling
        if result.lower().startswith("true"):
            st.success(f"✅ **Verdict:** {result}")
            st.balloons()
        elif result.lower().startswith("false"):
            st.error(f"❌ **Verdict:** {result}")
            st.snow()
        else:
            st.info(f"💡 **Verdict:** {result}")


st.divider()
st.caption("Disclaimer: This tool provides a verification based on the LLM's grounding against available sources and should not replace professional fact-checking.")
st.caption("Developed by:")
st.markdown("[Adarsh Yadav](https://www.github.com/yadavadarsh55)")