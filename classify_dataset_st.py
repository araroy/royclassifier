import openai
import pandas as pd

def set_openai_client(api_key=None):
    """
    Initialize OpenAI client with the provided API key.
    If API key is None, fetch it from the environment or Streamlit secrets.
    """
    if not api_key:
        raise ValueError("OpenAI API Key not provided.")
    openai.api_key = api_key
    return openai

def classify_text(client, prompt, text):
    """
    Sends a single text input to the OpenAI API for classification using the given prompt.
    """
    messages = [
        {"role": "system", "content": f"You are an assistant trained to classify data based on the following prompt: {prompt}"},
        {"role": "user", "content": f"Text: {text}"}
    ]
    try:
        response = client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

def classify_dataset(client, data, column_to_classify, classification_prompt):
    """
    Classify each row in the specified column of the dataset using OpenAI's API.
    """
    if column_to_classify not in data.columns:
        raise ValueError(f"Column '{column_to_classify}' not found in the dataset.")

    # Apply classification row by row
    data['Label'] = data[column_to_classify].apply(
        lambda x: classify_text(client, classification_prompt, x) if pd.notna(x) else "Error"
    )
    return data
