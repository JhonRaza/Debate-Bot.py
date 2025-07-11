import streamlit as st
import sys
sys.path.append('src')
from src.fact_checking import fact_check_query, query_huggingface_api
from src.retrieval import populate_chromadb
import warnings
warnings.simplefilter("ignore")


st.title("🧐 Fact-Checking System (RAG-Based, LLM Inference API implementation)")

# Sidebar for configuring the model
st.sidebar.title("Model Settings")
# st.sidebar.text("🔗 Enter your Huggingface API Token")
# token = st.sidebar.text_input("API Token")
# button = st.sidebar.button("Save API Token")
# if button:
#     st.session_state.HUGGINGFACEHUB_API_TOKEN = token
#     print(st.session_state.HUGGINGFACEHUB_API_TOKEN)
#     st.sidebar.success("API Token saved successfully!")

st.sidebar.write("Configure the model settings for text generation.")
cosine_distance_threshold = st.sidebar.slider("Cosine Distance", min_value=0.0, max_value=2.0, value=0.85, step=0.05)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=200, step=10)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

model_options = [
    "llama3.1"
]
columns = st.columns([2,2])
with columns[0]:
    selected_model = st.selectbox("Select a model for text generation:", model_options)


if "fact_check_result" not in st.session_state:
    st.session_state.fact_check_result = None
if "similar_claim" not in st.session_state:
    st.session_state.similar_claim = None
if "recheck_result" not in st.session_state:
    st.session_state.recheck_result = None
if "user_query" not in st.session_state:
    st.session_state.user_query = None


user_query = st.text_input("Enter a claim to fact-check:", placeholder="Write a short story about hobbits...")

if st.button("Check Statement"):
    # if "HUGGINGFACEHUB_API_TOKEN" in st.session_state:
    # print(st.session_state.HUGGINGFACEHUB_API_TOKEN)
    with st.spinner("Fact checking..."):
        try:
            if user_query != "":
                response, similar_claim = fact_check_query(user_query, selected_model, max_tokens, temperature, cosine_distance_threshold)
                st.session_state.fact_check_result = response
                st.session_state.similar_claim = similar_claim
                st.session_state.user_query = user_query
                st.session_state.recheck_result = None  # Reset the recheck result when a new query is entered 
            else:
                st.warning("⚠️ Please enter a claim to fact-check.")
                st.session_state.fact_check_result = None
                st.session_state.similar_claim = None
                st.session_state.user_query = None
                st.session_state.recheck_result = None 
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    # else:
    #     st.warning("⚠️ Please enter your Huggingface API Token in the sidebar to proceed.")
if st.session_state.fact_check_result:
    response = st.session_state.fact_check_result
    similar_claim = st.session_state.similar_claim
    user_query = st.session_state.user_query
    st.write("🔍 **User Query:**", st.session_state.user_query)
    if similar_claim:
        st.write("**AI-powered Fact-Check Result:**", response)
        st.divider()
        st.write(f"🔍 **Similar Claim found in the corpus:** *{similar_claim['statement']}*")
        st.write(f"📊 **Label:** {similar_claim['label']}")
    else:
        st.write("⚠️ **No texts backing up your claim were found in the corpus. Would you like to use the LLM to rate this claim?**")
        
        if st.button("Rate this claim"):
            with st.spinner("Fact checking..."):
                try:
                    recheck_response, _ = query_huggingface_api(user_query, selected_model, max_tokens, temperature, None)
                    st.session_state.recheck_result = recheck_response
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if st.session_state.recheck_result:
    st.write("**LLM-based Fact-Check Result:**", st.session_state.recheck_result)
