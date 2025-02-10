from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGooglePalm

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
        chain = create_stuff_documents_chain(llm=model, prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing conversational chain: {e}")
        logging.error(f"Error details: {e}")
        return None
