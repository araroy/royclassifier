import pandas as pd
import openai
import time
from tqdm import tqdm
import streamlit as st

def set_openai_client():
    """
    Initialize and return the OpenAI client with the API key from Streamlit secrets.
    """
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key not found. Please add it to Streamlit secrets.")
    
    # Initialize the OpenAI client
    return openai.Client(api_key=api_key)

def classify_text(client, prompt, text):
    """
    Sends a single text input to the OpenAI API for classification using a client instance.
    """
    messages = [
        {"role": "system", "content": f"You are an assistant trained to classify data based on the following prompt: {prompt}"},
        {"role": "user", "content": f"Text: {text}"}
    ]
    try:
        # Use client to call the chat completions API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        # Extract the content
        label = response.choices[0].message.content

        return label
    except Exception as e:
        return f"Error: {e}"

def classify_dataset(client, data, column_to_classify, classification_prompt, batch_size=100):
    """
    Processes a dataset in memory by classifying the specified column.
    """
    # Ensure the column exists
    if column_to_classify not in data.columns:
        raise ValueError(f"Column '{column_to_classify}' not found in the dataset.")

    data[column_to_classify] = data[column_to_classify].fillna("").str.strip()
    total_rows = len(data)

    labels = []
    for i in tqdm(range(0, total_rows, batch_size), desc="Classifying"):
        batch = data[column_to_classify][i:i + batch_size].tolist()
        try:
            batch_labels = [classify_text(client, classification_prompt, text) for text in batch]
            labels.extend(batch_labels)
        except Exception as e:
            labels.extend(["Error"] * len(batch))
            print(f"Batch {i} failed with error: {e}")
        
        time.sleep(2)  # Introduce a longer delay after each batch

    # Final check for length mismatch
    if len(labels) != total_rows:
        labels.extend(["Error"] * (total_rows - len(labels)))

    data['Label'] = labels
    return data
