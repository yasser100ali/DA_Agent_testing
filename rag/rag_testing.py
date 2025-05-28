import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time 
import utils 
import streamlit as st

st.text_area('hi')

# Function to load documents from a file
def load_documents(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

a = time.time()
# Load documents
documents = load_documents('agents_doc.txt')
b = time.time()
st.write(f'Time taken to load documents: {round(b-a, 3)} seconds')

# Initialize sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

a = time.time()
# Generate and normalize document embeddings
doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
b = time.time()
st.write(f'Generating and normalizing document space: {round(b-a, 3)} seconds.')


# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

# Function to retrieve documents using FAISS
def retrieve_documents(query, k=1):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
    similarities, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return retrieved_docs

# Function to perform RAG query
def rag_query(query, k=1, model="deepseek-chat"):
    a = time.time()
    retrieved_docs = retrieve_documents(query, k)
    b = time.time()

    print(f'Retrieval finished in {b-a} seconds.')

    context = "\n\n".join(retrieved_docs)
    print(f'\nStart of context\n{context}\nEnd of context.\n')


    user_prompt = f"Question: {query}\n\nContext: {context}\n\nAnswer:"
    system_prompt = "You are a helpful assistant."
    response = utils.get_response(system_prompt, user_prompt, model)
    full_response = utils.display_stream(response)
    return full_response

# Example usage
if __name__ == "__main__":
    
    query = "What is ACT-R? What section could I find this information on?"  # Adjust based on agents_doc.txt content
    st.write(f'Query: {query}')
    a = time.time()
    answer = rag_query(query, k=3)
    b = time.time()
