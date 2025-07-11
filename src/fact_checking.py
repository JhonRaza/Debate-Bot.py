import requests
from src.retrieval import retrieve_similar_claim
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from langchain_community.llms import Ollama
import warnings
warnings.simplefilter("ignore")

import os
from dotenv import load_dotenv

load_dotenv()




def query_huggingface_api(formatted_input, selected_model, max_tokens, temperature, similar_claim):
    # Set your Hugging Face API key
    # hf_api_key = st.session_state["HUGGINGFACEHUB_API_TOKEN"]
    # llm = HuggingFaceEndpoint(
    #         repo_id=selected_model,
    #         model_kwargs={"max_length": max_tokens},
    #         huggingfacehub_api_token=hf_api_key,
    #         temperature=temperature,
    #         )
    llm = Ollama(model=selected_model)
    template = "You are a helpful assistant designed to answer a user's queries. Answer the following user query precisely: |query| {input_text} |assistant|: "
    prompt_template = PromptTemplate(template=template, input_variables=["input_text"])

    chain = LLMChain(llm=llm, prompt=prompt_template)

    generated_text = chain.invoke({ "input_text": formatted_input }, verbose=True)
    return generated_text['text'], similar_claim

def query_huggingface_api_cli(formatted_input, selected_model, max_tokens, temperature, similar_claim):
    # hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # if not hf_api_key:
    #     raise ValueError("Error: HUGGINGFACEHUB_API_TOKEN environment variable is not set. Please set it before running.")


    # llm = HuggingFaceEndpoint(
    #         repo_id=selected_model,
    #         model_kwargs={"max_length": max_tokens},
    #         huggingfacehub_api_token=hf_api_key,
    #         temperature=temperature,
    #         )

    llm = Ollama(model=selected_model)
    template = "You are a helpful assistant designed to answer a user's queries. Answer the following user query precisely: |query| {input_text} |assistant|: "
    prompt_template = PromptTemplate(template=template, input_variables=["input_text"])

    chain = LLMChain(llm=llm, prompt=prompt_template)

    generated_text = chain.invoke({ "input_text": formatted_input }, verbose=True)
    return generated_text['text'], similar_claim

def fact_check_query(query, selected_model, max_tokens, temperature, threshold):
    similar_claim = retrieve_similar_claim(query, threshold)
    
    if not similar_claim:
        # formatted_input = f"User query: {query}\n\nThis claim has not been fact-checked before. Provide an analysis based on general knowledge."
        return "NOPEE", None


    else:
        print(similar_claim)
        formatted_input = f"User's query: {query}\n\nClosest fact-checked claim: {similar_claim['statement']}\nThis claim was labelled: {similar_claim['label']}\n\nBased on this information and using it, generate a fact-checking response."
        return query_huggingface_api(formatted_input, selected_model, max_tokens, temperature, similar_claim)

def fact_check_query_cli(query, selected_model, max_tokens, temperature, threshold):
    similar_claim = retrieve_similar_claim(query, threshold)
    
    if not similar_claim:
        # formatted_input = f"User query: {query}\n\nThis claim has not been fact-checked before. Provide an analysis based on general knowledge."
        return "NOPEE", None


    else:
        print(similar_claim)
        formatted_input = f"User's query: {query}\n\nClosest fact-checked claim: {similar_claim['statement']}\nThis claim was labelled: {similar_claim['label']}\n\nBased on this information and using it, generate a fact-checking response."
        return query_huggingface_api_cli(formatted_input, selected_model, max_tokens, temperature, similar_claim)
