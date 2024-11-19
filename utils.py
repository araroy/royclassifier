import streamlit as st
import openai

def set_openai_key():
    """Fetch the OpenAI API key from Streamlit Secrets and set it globally."""
    openai.api_key = st.secrets["OPENAI_API_KEY"]
