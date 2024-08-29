import streamlit as st
import PyPDF2
from groq import Groq
import io
import os
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

os.environ['GROQ_API_KEY'] = st.secrets['groq_api_key']

# Function to summarize text using Groq LLM
def summarize_text_with_groq(extracted_text):
    client = Groq()
    prompt = f"Summarize the following text from a research in a concise manner: {extracted_text}"
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=5000,
        top_p=1,
        stream=True,
        stop=None,
    )
    summary = ""
    for chunk in completion:
        summary += chunk.choices[0].delta.content or ""
    return summary

# Streamlit app
def main():
    st.title("PDF Summarizer using Groq LLM")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))

        # Display extracted text
        st.subheader("Extracted Text")
        st.text_area("", text, height=200)

        # Summarize button
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarize_text_with_groq(text)
            
            # Display summary
            st.subheader("Summary")
            st.write(summary)

if __name__ == "__main__":
    main()