import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

genai.configure(api_key=google_api_key)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get embeddings from Gemini API
def get_gemini_embeddings(text):
    try:
        response = genai.generate_embeddings(
            model="models/embedding-001",  # Replace with actual model if necessary
            text=text
        )
        return response.embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = [get_gemini_embeddings(chunk) for chunk in text_chunks]
    embeddings = [embedding for embedding in embeddings if embedding]  # Filter out None values
    vector_store = FAISS.from_embeddings(embeddings)
    vector_store.save_local("faiss_index")

# Function to generate a question-answering chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: 'Answer is not available in the context.'
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response.get("output_text", "No response available"))
    except Exception as e:
        st.error(f"Error while processing user input: {e}")

# Streamlit main function
def main():
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
