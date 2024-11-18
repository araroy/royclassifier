import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from classify_dataset_st import classify_dataset  # Replace with the actual script/module name

# Title and Description
st.title("Text Classification and Visualization Portal")
st.markdown("""
Upload an Excel file, classify text, and visualize the results.
""")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Load dataset
    df = pd.read_excel(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(df.head())

    # Column selection
    column_to_classify = st.selectbox("Select the column to classify", options=df.columns)

    # Classification prompt
    classification_prompt = st.text_area(
        "Enter your classification prompt",
        "Classify the column as follows:\n"
        "1. Decision maker: if the writer has made any consumption-related decisions (e.g., which restaurant to visit, what food to order) for others.\n"
        "2. Participant: all other reviews."
    )

    if st.button("Run Classification"):
        # Run the classification
        st.info("Running classification...this might take some time.")
        try:
            classified_data = classify_dataset(df, column_to_classify, classification_prompt)
            st.success("Classification completed successfully!")

            # Display classified data
            st.write("Classified Data:")
            st.write(classified_data.head())

            # Download option
            output = BytesIO()
            classified_data.to_excel(output, index=False, engine='openpyxl')
            st.download_button(
                label="Download Classified Data",
                data=output.getvalue(),
                file_name="classified_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Visualization
            st.markdown("### Visualizations")
            label_counts = classified_data['Label'].value_counts()

            # Bar Chart for Label Distribution
            st.markdown("**Label Distribution**")
            fig, ax = plt.subplots()
            label_counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title("Distribution of Labels")
            ax.set_xlabel("Labels")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
