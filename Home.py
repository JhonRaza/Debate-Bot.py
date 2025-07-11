import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fact-Checking App", layout="centered")
    
# Title and Subtitle
st.title("Fact-Checking App: AI-Powered Verification")
st.subheader("Combating Misinformation with AI")

# Introduction
st.write(
    "Misinformation spreads rapidly in today's digital world. This **AI-powered Fact-Checking App** leverages "
    "**Retrieval-Augmented Generation (RAG)** to verify claims against fact-checked data, providing accurate "
    "and detailed responses."
)

# How It Works Section
st.subheader("📌 How It Works")

st.markdown("**1️⃣ Claim Retrieval**")
st.write(
    "- The system searches for **relevant fact-checked claims** in a knowledge base using **ChromaDB**."
    "\n- It ranks the most similar claims using **cosine similarity**."
)

st.markdown("**2️⃣ AI-Powered Fact-Checking**")
st.write(
    "- A **Hugging Face Large Language Model (LLM)**, powered by **LangChain**, analyzes the retrieved facts."
    "\n- It **generates a contextual response**, explaining whether the claim aligns with known facts or appears misleading."
)

st.markdown("**3️⃣ Handling New or Ambiguous Claims**")
st.write(
    "- If no exact match is found, the LLM **uses reasoning and available knowledge** to assess the statement."
    "\n- It highlights **uncertainties and possible sources of verification**."
)

# Tech Stack Section
st.subheader("🛠 Tech Stack")
st.markdown(
    "- **LLM Pipeline:** LangChain + Hugging Face Inference API\n"
    "- **Database:** ChromaDB for efficient fact retrieval\n"
    "- **Similarity Search:** Cosine similarity for accurate claim matching\n"
    "- **Frontend:** Streamlit for an intuitive user experience"
)

# Call to Action
st.subheader("🚀 Get Started!")
st.write("Enter a claim and let AI verify it for you!")

# Developer Info
st.write("Developed by **Jhon Raza**")