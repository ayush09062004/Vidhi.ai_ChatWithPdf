# Import necessary libraries
from duckduckgo_search import DDGS
from groclake.modellake import ModelLake
import logging
import os
from PyPDF2 import PdfReader
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set up Modellake
os.environ['GROCLAKE_API_KEY'] = "Your API Key"  # Replace with your Grocklake API key
os.environ['GROCLAKE_ACCOUNT_ID'] = "Your Account Id"  # Replace with your Grocklake account ID
modellake = ModelLake()

# Load a pre-trained sentence transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for vector search
dimension = 384  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

# Dictionary to store text chunks and their corresponding embeddings
knowledge_base = {}

def load_knowledge_base(pdf_files):
    """
    Load text from PDF files, split into chunks, and store embeddings in FAISS index.
    """
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # Split text into smaller chunks (e.g., sentences or paragraphs)
            chunks = text.split(". ")  # Splitting by sentences for simplicity
            for chunk in chunks:
                if chunk.strip():  # Ignore empty chunks
                    # Generate embedding for the chunk
                    embedding = embedding_model.encode([chunk])[0]
                    # Add embedding to FAISS index
                    index.add(np.array([embedding]))
                    # Store chunk in knowledge base with its index
                    knowledge_base[len(knowledge_base)] = {"source": pdf_file, "text": chunk}
        except Exception as e:
            print(f"Error reading {pdf_file}: {str(e)}")

# List of PDF files
pdf_files = ["BNS.pdf", "BNSS.pdf", "BSA.pdf", "COI.pdf"]

# Load knowledge base
load_knowledge_base(pdf_files)

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
        print(f"Error during DuckDuckGo search: {str(e)}")
        return None

# Vector Search in Knowledge Base
def vector_search_knowledge_base(query, top_k=3):
    """
    Perform a vector search in the knowledge base for relevant information.
    """
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query])[0]
        # Search FAISS index for the top-k most similar embeddings
        distances, indices = index.search(np.array([query_embedding]), top_k)
        # Retrieve the corresponding text chunks
        results = []
        for idx in indices[0]:
            if idx != -1:  # Ignore invalid indices
                results.append(knowledge_base[idx])
        return results if results else None
    except Exception as e:
        print(f"Error during vector search: {str(e)}")
        return None

# Summarize Text Function
def summarize_text(text, max_tokens=500):
    """
    Summarize the text to fit within the specified token limit.
    """
    sentences = text.split(". ")
    summarized_text = ". ".join(sentences[:5])  # Adjust the number of sentences as needed
    return summarized_text[:max_tokens]

# Generate Concise Summary Using Modellake
def generate_summary(context, query):
    """
    Generate a concise summary of the context using Modellake.
    """
    system_prompt = (
        "You are a legal assistant having specialization in Indian law. Brief the following context in a very concise manner, retaining all key points relevant to the user's query. Ensure the it is very brief and fits within the token limit."
    )
    chat_completion_request = {
        "groc_account_id": os.environ['GROCLAKE_ACCOUNT_ID'],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ],
        "max_tokens": 500  # Set a token limit for the summary
    }
    try:
        response = modellake.chat_complete(chat_completion_request)
        print("Summary API Response:", response)  # Debugging: Print the full API response
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        elif "answer" in response:
            return response["answer"]
        else:
            return context[:500]  # Fallback to truncation if summarization fails
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return context[:500]  # Fallback to truncation if summarization fails

# Chatbot Function without Chat History
def chatbot(query):
    """
    Generate a response to the user's query using the knowledge base, DuckDuckGo search results, and Modellake.
    """
    try:
        # Search the knowledge base first using vector search
        kb_results = vector_search_knowledge_base(query)
        if kb_results:
            context = "\n".join([result["text"] for result in kb_results])
        else:
            # Fall back to DuckDuckGo search
            search_results = duckduckgo_search(query)
            if not search_results:
                return "I couldn't find any relevant information. Please try rephrasing your query."
            # Combine search results into a single context (with a limit)
            max_context_length = 500  # Adjust based on token limit
            context = "\n".join([result["body"][:200] for result in search_results])[:max_context_length]

        # Generate a concise summary of the context
        summarized_context = generate_summary(context, query)

        # Use Modellake for response generation
        system_prompt = (
            "You are a legal assistant having specialization in Indian law. Provide brief, accurate and concise legal advice without missing any key detail in strictly 100-120 words. Focus only on Indian laws, rules, and regulations. Ensure your responses are brief, to the point, and do not truncate any key details. Note: The Indian Penal Code (IPC) has been replaced by the Bharatiya Nyaya Sanhita (BNS), the Code of Criminal Procedure (CrPC) by the Bharatiya Nagrik Suraksha Sanhita (BNSS), and the Indian Evidence Act by the Bharatiya Sakshya Adhiniyam (BSA)."
        )

        chat_completion_request = {
            "groc_account_id": os.environ['GROCLAKE_ACCOUNT_ID'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {summarized_context}\n\nQuestion: {query}"}
            ],
            "max_tokens": 4086  # Set the max tokens for the response
        }

        response = modellake.chat_complete(chat_completion_request)
        print("Chat API Response:", response)  # Debugging: Print the full API response
        if "choices" in response and len(response["choices"]) > 0:
            full_response = response["choices"][0]["message"]["content"]
        elif "answer" in response:
            full_response = response["answer"]
        else:
            full_response = "Unable to generate a complete response."

        return full_response
    except Exception as e:
        print(f"Error in chatbot function: {str(e)}")  # Debugging: Print the error
        return f"Error generating response: {str(e)}"

# Gradio Chat Interface
def gradio_chatbot(query, chat_history):
    """
    Gradio chatbot function that ignores chat history and treats each query independently.
    """
    response = chatbot(query)
    return response

# Create Gradio Chat Interface
iface = gr.ChatInterface(
    fn=gradio_chatbot,
    title="Vidhi.ai: Indian Legal Assistance Agent",
    description="Ask any legal question related to Indian law, and I'll try to provide concise advice.",
    chatbot=gr.Chatbot(height=400)  # Adjust height as needed
)

# Launch the Gradio Interface
if __name__ == "__main__":
    iface.launch(share=True)
