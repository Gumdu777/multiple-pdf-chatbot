# Multiple PDF Chatbot

A streamlined solution for querying across multiple PDF documents using an intelligent chatbot interface. This project leverages natural language processing (NLP) capabilities powered by Google Generative AI, efficient vector storage through FAISS, and a user-friendly web interface built with Streamlit. 

## Project Overview

The **Multiple PDF Chatbot** enables users to interact with multiple PDF documents by asking natural language questions. This chatbot provides a precise response by utilizing vector embeddings for efficient document search and Google Generative AI for question understanding. It is ideal for anyone needing rapid insights from large PDF collections, such as researchers, analysts, and legal professionals.

## Features

- **Natural Language Questioning**: Ask questions in plain English, and the chatbot retrieves relevant information directly from the PDFs.
- **Multiple PDF Support**: Upload multiple PDFs, making it easy to query across various documents.
- **Efficient Document Search**: Uses FAISS for storing and querying vector embeddings, ensuring fast and accurate results.
- **Web-Based Interface**: Streamlit provides a simple, accessible web application interface for seamless user interaction.
- **Scalable Solution**: Suitable for large datasets of PDF documents due to the efficient FAISS-based vector store.

## Installation Guide

To get started with the Multiple PDF Chatbot, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Gumdu777/multiple-pdf-chatbot.git
   cd multiple-pdf-chatbot
   ```

2. **Set Up a Virtual Environment**  
   It's recommended to use a virtual environment to manage dependencies.
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Google API Key**  
   Add your Google Generative AI API key to a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run the Application**
   ```bash
   streamlit run main.py
   ```

## Detailed Usage Instructions

1. **Upload PDFs**  
   Use the "Upload PDF" button to upload one or more PDF documents.

2. **Ask Questions**  
   Type questions in the input box. The chatbot will analyze your question, search through the PDF content, and return the most relevant response.

3. **View Results**  
   The answer to each query is displayed directly in the chat window. You can also review which document sections were referenced to generate the response.

## Technology Stack

- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Vector Database**: FAISS (Facebook AI Similarity Search) for efficient document embedding and retrieval
- **NLP Model**: Google Generative AI for understanding and responding to user queries
- **Environment Management**: Python virtual environments
- **APIs**: Google Generative AI API for NLP capabilities

## Future Enhancements

Planned improvements for this project include:

- **Enhanced Multi-Language Support**: Extend NLP capabilities to support queries in multiple languages.
- **Integration with Additional Document Types**: Extend compatibility beyond PDF to other document types (e.g., DOCX, TXT).
- **User Authentication**: Implement user authentication for access control and personalized document interaction.
- **Improved Query Accuracy**: Fine-tune the FAISS embedding search to better handle ambiguous queries.

## Author and Contact Information

Created by [Tejas Kudale](https://www.linkedin.com/in/tejas-kudale-8854812b3).

For questions or collaboration inquiries, feel free to reach out via GitHub or LinkedIn.

---

