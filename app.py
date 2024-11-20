import streamlit as st
import pandas as pd
from io import BytesIO
from classify_dataset_st import classify_dataset, set_openai_client

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
        st.stop()

    # Column selection
    column_to_classify = st.selectbox("Select the column to classify", options=df.columns)
    if not pd.api.types.is_string_dtype(df[column_to_classify]):
        st.error("The selected column is not a text column. Please select a valid text column.")
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
        st.stop()

    if st.button("Run Classification"):
        st.info("Running classification...this might take some time.")
        try:
            # Initialize OpenAI client
            api_key = st.secrets["OPENAI_API_KEY"]
            client = set_openai_client(api_key)

            # Run classification
            classified_data = classify_dataset(
                client=client,
                data=df,
                column_to_classify=column_to_classify,
                classification_prompt=classification_prompt
            )
            st.success("Classification completed successfully!")

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
