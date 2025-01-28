import os
import logging
from duckduckgo_search import DDGS
from groclake.modellake import ModelLake
from pypdf import PdfReader  # Use pypdf instead of PyPDF2
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Modellake
os.environ['GROCLAKE_API_KEY'] = "202cb962ac59075b964b07152d234b70"  # Replace with your Grocklake API key
os.environ['GROCLAKE_ACCOUNT_ID'] = "2466cc1f4e5446effb7a9250de957141"  # Replace with your Grocklake account ID
modellake = ModelLake()

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logger.info("Successfully extracted text from PDF.")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        return None

# Function to split text into chunks for retrieval
def split_text_into_chunks(text, chunk_size=500):
    """
    Split text into smaller chunks for retrieval.
    """
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to retrieve relevant chunks from the PDF
def retrieve_relevant_chunks(query, chunks):
    """
    Retrieve relevant chunks from the PDF based on the query.
    """
    relevant_chunks = []
    for chunk in chunks:
        if query.lower() in chunk.lower():  # Basic keyword matching
            relevant_chunks.append(chunk)
    return relevant_chunks

# DuckDuckGo Search Function
def duckduckgo_search(query, max_results=3):
    """
    Perform a web search using DuckDuckGo and return the top results.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        return results if results else None
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search: {str(e)}")
        return None

# Function to summarize the document
def summarize_document(text):
    """
    Summarize the document using the LLM.
    """
    system_prompt = (
        "You are a legal expert assistant. Very Briefly Explain the following legal document in a concise and informative way in strictly 100-120 words, focusing on key legal concepts, terms, and provisions."
    )
    chat_completion_request = {
        "groc_account_id": os.environ['GROCLAKE_ACCOUNT_ID'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document: {text}"}
        ],
        "max_tokens": 500  # Set a token limit for the summary
    }
    try:
        response = modellake.chat_complete(chat_completion_request)
        logger.info(f"Summary API Response: {response}")
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        elif "answer" in response:
            return response["answer"]
        else:
            return text[:500]  # Fallback to truncation if summarization fails
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return text[:500]  # Fallback to truncation if summarization fails

# Function to generate answers from web search results
def generate_answer_from_web_search(query, search_results):
    """
    Generate an answer from DuckDuckGo search results using the LLM.
    """
    system_prompt = (
        "You are a legal expert assistant. Use the following web search results to answer the user's question. Provide a very brief, concise and accurate response based on the information available within 100-120 words."
    )
    context = "\n".join([result.get("title", "") + ": " + result.get("body", "") for result in search_results])
    chat_completion_request = {
        "groc_account_id": os.environ['GROCLAKE_ACCOUNT_ID'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
        ],
        "max_tokens": 500  # Set a token limit for the response
    }
    try:
        response = modellake.chat_complete(chat_completion_request)
        logger.info(f"Web Search Answer API Response: {response}")
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        elif "answer" in response:
            return response["answer"]
        else:
            return "Unable to generate a response from web search results."
    except Exception as e:
        logger.error(f"Error generating answer from web search: {str(e)}")
        return "Unable to generate a response from web search results."

# Function to handle PDF upload and generate summary
def handle_pdf_upload(pdf_file):
    """
    Handle PDF upload and generate a summary.
    """
    if pdf_file is None:
        return "Please upload a PDF file first."

    pdf_text = extract_text_from_pdf(pdf_file)
    if not pdf_text:
        return "Error: Unable to extract text from the uploaded PDF."

    summary = summarize_document(pdf_text)
    return f"Here's a summary of the document:\n\n{summary}"

# Function to handle user questions
def handle_user_question(pdf_file, query):
    """
    Handle user questions about the uploaded PDF.
    """
    if pdf_file is None:
        return "Please upload a PDF file first."

    pdf_text = extract_text_from_pdf(pdf_file)
    if not pdf_text:
        return "Error: Unable to extract text from the uploaded PDF."

    # Split text into chunks for retrieval
    chunks = split_text_into_chunks(pdf_text)

    # Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(query, chunks)

    # Augment with LLM to generate a response
    if relevant_chunks:
        context = "\n".join(relevant_chunks)
        system_prompt = (
            "You are a legal expert assistant having specialization in Indian laws. Answer the user's question in Indian context only based on the provided legal document in 100-120 words. It should be concise in such a manner that you don't miss out any key details. "
            "If the document does not contain the answer, say 'I don't know'."
        )
        chat_completion_request = {
            "groc_account_id": os.environ['GROCLAKE_ACCOUNT_ID'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document: {context}\n\nQuestion: {query}"}
            ],
            "max_tokens": 4086  # Set the max tokens for the response
        }

        # Get the LLM's response
        response = modellake.chat_complete(chat_completion_request)
        logger.info(f"LLM Response: {response}")

        if "choices" in response and len(response["choices"]) > 0:
            llm_answer = response["choices"][0]["message"]["content"]
        elif "answer" in response:
            llm_answer = response["answer"]
        else:
            llm_answer = "Unable to generate a response."

        # If the LLM doesn't know, fallback to DuckDuckGo search
        if "I don't know" in llm_answer or "Unable to generate" in llm_answer:
            search_results = duckduckgo_search(query)
            if not search_results:
                return "I couldn't find any relevant information in the PDF or online. Please try rephrasing your query."

            # Generate an answer from web search results
            web_answer = generate_answer_from_web_search(query, search_results)
            return f"{web_answer}"
        else:
            return llm_answer
    else:
        # If no relevant chunks are found, perform a DuckDuckGo search
        search_results = duckduckgo_search(query)
        if not search_results:
            return "I couldn't find any relevant information in the PDF or online. Please try rephrasing your query."

        # Generate an answer from web search results
        web_answer = generate_answer_from_web_search(query, search_results)
        return f"{web_answer}"

# Gradio Chat Interface
def gradio_chat_interface(pdf_file, query, chat_history):
    """
    Gradio chat interface function to handle PDF upload and chat.
    """
    if pdf_file is not None and (chat_history is None or len(chat_history) == 0):
        # Automatically generate a summary when the PDF is uploaded
        summary = handle_pdf_upload(pdf_file)
        chat_history.append(("System", summary))
        chat_history.append(("System", "You can now ask questions about the document."))
        return chat_history, ""  # Clear the query input box

    if query is not None and query.strip() != "":
        # Handle user questions
        response = handle_user_question(pdf_file, query)
        chat_history.append(("User", query))
        chat_history.append(("System", response))
        return chat_history, ""  # Clear the query input box

    return chat_history, ""  # Clear the query input box

# Main function to launch the Gradio app
def main():
    # Create Gradio chat interface
    with gr.Blocks() as demo:
        # PDF upload component
        pdf_file = gr.File(label="Upload PDF File")

        # Chatbot component
        chatbot = gr.Chatbot(label="Chat with Your Legal PDF")

        # Textbox for user questions
        query = gr.Textbox(label="Ask a Question About the PDF", placeholder="Type your question here...")

        # Chat history state
        chat_history = gr.State([])

        # Submit button
        submit_button = gr.Button("Submit")

        # Define the interaction
        submit_button.click(
            gradio_chat_interface,
            inputs=[pdf_file, query, chat_history],
            outputs=[chatbot, query]
        )

        # Enable Enter key to submit
        query.submit(
            gradio_chat_interface,
            inputs=[pdf_file, query, chat_history],
            outputs=[chatbot, query]
        )

        # Automatically trigger summary generation when a PDF is uploaded
        pdf_file.change(
            gradio_chat_interface,
            inputs=[pdf_file, query, chat_history],
            outputs=[chatbot, query]
        )

    # Launch Gradio app
    demo.launch(share=True)

# Entry point for Hugging Face Spaces
if __name__ == "__main__":
    main()