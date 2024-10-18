import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables and configure API
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

genai.configure(api_key=google_api_key)

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to get Google Gemini embeddings for a piece of text
def get_gemini_embeddings(text: str):
    """Call Gemini API to get embeddings for a piece of text."""
    response = genai.generate_embeddings(
        model="models/embedding-001",  # Replace with your actual Gemini model for embeddings
        text=text
    )
    return response.embeddings

# Function to create a vector store using FAISS and Google Gemini embeddings
def get_vector_store(text_chunks):
    embeddings = [get_gemini_embeddings(chunk) for chunk in text_chunks]
    vector_store = FAISS.from_embeddings(embeddings, texts=text_chunks)
    vector_store.save_local("faiss_index")

# Function to set up a conversational chain for answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: 'Answer is not available in the context.'
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    # Using Gemini's chat model for responses (can be adjusted if needed)
    model = genai.Chat(model="gemini-pro")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return ConversationalRetrievalChain.from_chain(model=model, retriever=FAISS.load_local("faiss_index").as_retriever(), prompt=prompt)

# Function to process user input and return a response
def user_input(user_question):
    embeddings = get_gemini_embeddings
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    # Use the conversational chain to answer the question
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response.get("output_text", "No response available"))

# Main Streamlit app
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="💁")
    st.header("Chat with Multiple PDFs using Gemini")
    
    user_question = st.text_input("Ask a question from the PDF files:")
    
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
