import pandas as pd
import openai
import os
import time
from tqdm import tqdm
from pathlib import Path
import glob

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set
=

if api_key:
    print("API Key loaded successfully.")
else:
    print("API Key not found. Please set the OPENAI_API_KEY environment variable.")


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
        print(f"Error processing text: {text[:30]}... | Error: {e}")
        return "Error"

def process_dataset(input_path, output_path, column_to_classify, classification_prompt, batch_size=100):
    """
    Processes a dataset by classifying the specified column, working with file paths.
    """
    data = pd.read_excel(input_path, engine='openpyxl')

    # Ensure the column exists
    if column_to_classify not in data.columns:
        raise ValueError(f"Column '{column_to_classify}' not found in the dataset.")

    data[column_to_classify] = data[column_to_classify].fillna("").str.strip()
    total_rows = len(data)

    labels = []
    for i in tqdm(range(0, total_rows, batch_size), desc=f"Processing {Path(input_path).name}"):
        batch = data[column_to_classify][i:i + batch_size].tolist()
        batch_labels = []
        for text in batch:
            label = classify_text(classification_prompt, text)
            batch_labels.append(label)
            time.sleep(1)
        labels.extend(batch_labels)

    if len(labels) != total_rows:
        print(f"Mismatch in labels: Expected {total_rows}, got {len(labels)}. Filling with 'Error'.")
        labels.extend(["Error"] * (total_rows - len(labels)))

    data['Label'] = labels
    data.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Classified dataset saved to '{output_path}'")

if __name__ == "__main__":
    # Define input and output folders
    input_folder = "input"
    output_folder = "output"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define classification prompt and column to classify
    column_to_classify = "Review"  # Adjust based on your dataset
    classification_prompt = (
        "Classify the column as follows:\n"
        "1. Decision maker: if the writer has made any consumption-related decisions (e.g., which restaurant to visit, what food to order) for others.\n"
        "2. Participant: all other reviews."
    )

    # Loop through all Excel files in the input folder
    for input_file in glob.glob(f"{input_folder}/*.xlsx"):
        # Define output file path
        output_file = f"{output_folder}/{Path(input_file).stem}_output.xlsx"
        
        print(f"Processing file: {input_file} -> {output_file}")
        
        # Process the dataset
        process_dataset(input_file, output_file, column_to_classify, classification_prompt)

