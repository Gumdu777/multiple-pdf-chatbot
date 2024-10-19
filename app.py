import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.google_palm import GooglePalmEmbeddings  # Updated import
from langchain.chat_models import ChatGooglePalm  # Updated import
from langchain_community.vectorstores import FAISS  # Corrected import
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

# Initialize the Google Palm API using the provided key
# Assuming `genai` was being used to configure the API
# This setup is now done automatically within langchain when using GooglePalmEmbeddings or ChatGooglePalm

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates and saves vector store using FAISS and GooglePalmEmbeddings."""
    embeddings = GooglePalmEmbeddings(model="models/embedding-001")  # Updated to use GooglePalmEmbeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up a question-answering chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: 'Answer is not available in the context.'
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    model = ChatGooglePalm(model="gemini-pro", temperature=0.3)  # Updated to use ChatGooglePalm
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff"
