import streamlit as st
import pandas as pd
import openai

def set_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API Key not found in Streamlit secrets.")
    return openai.Client(api_key=api_key)

def classify_text(client, prompt, text):
    messages = [
        {"role": "system", "content": f"You are an assistant trained to classify data based on the following prompt: {prompt}"},
        {"role": "user", "content": f"Text: {text}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def classify_dataset(client, data, column_to_classify, classification_prompt, batch_size=10):
    if column_to_classify not in data.columns:
        raise ValueError(f"Column '{column_to_classify}' not found in the dataset.")

    labels = []
    for i in range(0, len(data), batch_size):
        batch = data[column_to_classify].iloc[i:i+batch_size]
        try:
            batch_labels = [classify_text(client, classification_prompt, text) for text in batch]
            labels.extend(batch_labels)
        except Exception as e:
            labels.extend(["Error"] * len(batch))
            print(f"Error with batch {i}: {e}")

    data['Label'] = labels
    return data

# Streamlit interface
st.title("Text Classifier")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(data.head())

    column_to_classify = st.selectbox("Select the column to classify", data.columns)
    classification_prompt = st.text_area("Enter the classification prompt", "Classify this text as decision-maker or participant based on if the writer has made any decision.")

    if st.button("Run Classification"):
        try:
            client = set_openai_client()
            classified_data = classify_dataset(client, data, column_to_classify, classification_prompt)
            st.write("Classified Data:")
            st.write(classified_data)
        except Exception as e:
            st.error(f"An error occurred: {e}")
