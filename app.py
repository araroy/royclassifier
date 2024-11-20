import streamlit as st
import pandas as pd
from io import BytesIO
from classify_dataset_st import classify_dataset, set_openai_client
import logging  # For debugging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("Application started...")

# Title and Description
st.title("Text Classification and Visualization Portal")
st.markdown("""
Upload an Excel file, classify text, and visualize the results.
""")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        logging.error(f"Error reading the file: {e}")
        st.stop()

    # Column selection
    column_to_classify = st.selectbox("Select the column to classify", options=df.columns)
    if not pd.api.types.is_string_dtype(df[column_to_classify]):
        st.error("The selected column is not a text column. Please select a valid text column.")
        logging.error("Selected column is not a valid text column.")
        st.stop()

    # Classification prompt
    classification_prompt = st.text_area(
        "Enter your classification prompt",
        "Classify the column as follows:\n"
        "1. Decision maker: if the writer has made any consumption-related decisions (e.g., which restaurant to visit, what food to order) for others.\n"
        "2. Participant: all other reviews."
    )

    if not classification_prompt.strip():
        st.error("Please provide a valid classification prompt.")
        logging.error("Classification prompt is empty.")
        st.stop()

    if st.button("Run Classification"):
        st.info("Running classification...this might take some time.")
        try:
            # Initialize OpenAI client
            api_key = st.secrets["OPENAI_API_KEY"]
            client = set_openai_client(api_key)
            logging.debug("OpenAI client initialized successfully.")

            # Check if the API key is accessible
            if not api_key:
                st.error("API Key not found. Please add your OpenAI API key to Streamlit secrets.")
                logging.error("API Key not found in Streamlit secrets.")

            else:
            # Mask most of the API key for security reasons and display the result
                st.success(f"API Key is accessible: {api_key[:5]}...********")
            # Test API connection
            try:
                test_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Test connection"}
                    ]
                )
                logging.debug(f"OpenAI test connection successful: {test_response}")
                st.success("OpenAI API connection verified successfully!")
            except Exception as api_test_error:
                st.error(f"Failed to verify OpenAI API connection: {api_test_error}")
                logging.error(f"API connection failed: {api_test_error}")
                st.stop()

            # Run classification
            classified_data = classify_dataset(
                client=client,
                data=df,
                column_to_classify=column_to_classify,
                classification_prompt=classification_prompt
            )
            st.success("Classification completed successfully!")
            logging.debug("Classification completed successfully.")

            # Display classified data
            st.write("Classified Data:")
            st.write(classified_data.head())

            # Download option
            output = BytesIO()
            classified_data.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(
                label="Download Classified Data",
                data=output.getvalue(),
                file_name="classified_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"An error occurred during classification: {e}")

