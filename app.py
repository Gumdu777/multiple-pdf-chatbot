import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import GooglePalmEmbeddings  # Updated import
from langchain_community.chat_models import ChatGooglePalm  # Updated import
from langchain_community.vectorstores import FAISS  # Corrected import
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging

# Load environment variables and configure API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle potential None return
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """Splits the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates and saves vector store using FAISS and GooglePalmEmbeddings."""
    try:
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logging.info("Initializing GooglePalmEmbeddings with model and API key.")
        
        # Pass the google_api_key when creating the embeddings instance
        embeddings = GooglePalmEmbeddings(model="models/embedding-001", google_api_key=google_api_key)  
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logging.info("Vector store generated successfully.")
    except Exception as e:
        st.error(f"Error generating vector store: {e}")
        logging.error(f"Error details: {e}")

def get_conversational_chain():
    """Sets up a question-answering chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say: 'Answer is not available in the context.'
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    try:
        model = ChatGooglePalm(model="gemini-pro", temperature=0.3)  # Updated to use ChatGooglePalm
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing conversational chain: {e}")
        logging.error(f"Error details: {e}")
        return None

def user_input(user_question):
    """Processes the user's question and returns a response from the vector store."""
    try:
        # Pass the google_api_key when creating the embeddings instance
        embeddings = GooglePalmEmbeddings(model="models/embedding-001", google_api_key=google_api_key)  
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            st.write("No relevant documents found for your question.")
            return
        
        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response.get("output_text", "No response available"))
    except Exception as e:
        st.error(f"Error processing user question: {e}")
        logging.error(f"Error details: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="💁")
    st.header("Chat with Multiple PDFs using Gemini 💁")
    
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
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing completed.")
                    else:
                        st.warning("No text found in the uploaded PDF files.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
