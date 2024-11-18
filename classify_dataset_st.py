import pandas as pd
import openai
import os
import time
from tqdm import tqdm
from pathlib import Path

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set

def classify_text(prompt, text):
    """
    Sends a single text input to the OpenAI API for classification.
    """
    messages = [
        {"role": "system", "content": f"You are an assistant trained to classify data based on the following prompt: {prompt}"},
        {"role": "user", "content": f"Text: {text}"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        label = response.choices[0].message.content.strip()
        return label
    except Exception as e:
        return "Error"

def classify_dataset(data, column_to_classify, classification_prompt, batch_size=100):
    """
    Processes a dataset in memory by classifying the specified column.
    """
    # Ensure the column exists
    if column_to_classify not in data.columns:
        raise ValueError(f"Column '{column_to_classify}' not found in the dataset.")

    data[column_to_classify] = data[column_to_classify].fillna("").str.strip()
    total_rows = len(data)

    labels = []
    for i in tqdm(range(0, total_rows, batch_size), desc=f"Classifying"):
        batch = data[column_to_classify][i:i + batch_size].tolist()
        batch_labels = [classify_text(classification_prompt, text) for text in batch]
        labels.extend(batch_labels)
        time.sleep(1)  # Rate limit

    if len(labels) != total_rows:
        labels.extend(["Error"] * (total_rows - len(labels)))

    data['Label'] = labels
    return data
