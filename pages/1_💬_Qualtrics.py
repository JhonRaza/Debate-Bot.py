import streamlit as st
from openai import OpenAI
import json

st.set_page_config(page_title="Qualtrics Debate Chat", page_icon="💬")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=   st.secrets["OPENAI_API_KEY"]  # Ensure you have set your OpenAI API key in secrets.toml
)

# System prompt for the debate bot
system_prompt = """You are partaking in an informal parliamentary-style debate (opening statements, rebuttals, closing statements).
You are debating *against* the position: The UK must increase its funding to nuclear energy. Start by inviting the user to make an opening statement.

To respond to the user, generate a <strong> response in a debate setting by following these steps:  
- Begin by listing your 10 strongest counterarguments against the user's position in concise bullet points.  
- Select only the 3 arguments ranked at the <top> of the list.  
- Write a concise, informal response (3-6 sentences) to the user's positions.
- Output should be in JSON format with three keys: "chain-of-thought reasoning" (the analysis of the user's arguments, reasoning, potential arguments, a ranked list of the arguments), "selected arguments" (the argument to be included in the response) and "response" (the short written response for the opponent).  
- Do not include any other explanation, meta-commentary, or extraneous formatting.

Output Format:  
Return a JSON object structured as:
{"chain-of-thought reasoning": "[reasoning and potential arguments]",
  "selected arguments": "[bullet point for 3 <top>-rank arguments]",
  "response": "[final response, 3-6 sentences]"
}"""

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]
help_prompt = """You are a debate assistant that helps users prepare for debates by JUST providing the user with relevant facts.
        DON'T help the user make an argument, just provide them with facts to back it up. Perform a web search to find relevant information and generate a response based on the user's input.
        """
if "help_messages" not in st.session_state:
    st.session_state.help_messages = [
        {"role": "system", "content": help_prompt}
    ]

st.title("💬 Qualtrics Debate Chat")
st.write("Debate *against* the position: **The UK must increase its funding to nuclear energy.**")

# Display chat history using Streamlit's chat_message
for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Chat input box
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare messages for OpenAI API
    messages = [
        {"role": "system", "content": system_prompt}
    ] + [
        m for m in st.session_state.messages[1:]
    ]
    # Display chat history using Streamlit's chat_message
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    bot_reply = response.choices[0].message.content
    # print(json.loads(bot_reply).get("response"))  # Print the JSON response for debugging
    reply = json.loads(bot_reply).get("response")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()


st.sidebar.header("Debate Assistant")
st.sidebar.write("Use this assistant to prepare for a debate by generating counterarguments and responses.")
help_call = st.sidebar.chat_input("Type your message here...", key="sidebar_input")
if help_call:
    st.session_state.help_messages.append({"role": "user", "content": help_call})
    # Prepare messages for OpenAI API
    help_messages = [
        {"role": "system", "content": help_prompt}
    ] + [
        m for m in st.session_state.help_messages[1:]
    ]
    # Get response from OpenAI
    help_response = client.chat.completions.create(
        model="gpt-4o-search-preview",
        messages=help_messages
    )
    help_reply = help_response.choices[0].message.content
    st.session_state.help_messages.append({"role": "assistant", "content": help_reply})
    print(help_reply)
    st.sidebar.write("**Assistant Response:**")
    for msg in st.session_state.help_messages[1:]:
        if msg["role"] == "user":
            with st.sidebar.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.sidebar.chat_message("assistant"):
                st.markdown(msg["content"])
    
    st.rerun()