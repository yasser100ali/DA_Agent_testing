from agents.agents import ReAct
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader  # Assuming PyPDF2 is installed; if not, pip install PyPDF2
import base64 
import streamlit as st 
from pdf2image import convert_from_bytes

class PDFDeepInsights:
    def __init__(self, user_input, local_var):
        self.user_input = user_input
        self.local_var = local_var
        # Initialize RAG components once in init for efficiency
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.index = None
        self._build_vectorstore()

    def _build_vectorstore(self):
        # Extract text from PDF pages
        reader = PdfReader(self.local_var["pdf_file_objects"])
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text().strip() if page.extract_text() else ""
            if text:  # Skip empty pages
                self.documents.append({'text': text, 'page': page_num + 1})  # 1-based page numbers

        if not self.documents:
            raise ValueError("No text extracted from the PDF.")

        # Generate and normalize document embeddings (one per page)
        doc_texts = [doc['text'] for doc in self.documents]
        doc_embeddings = self.embedding_model.encode(doc_texts)
        doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
        doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(doc_embeddings)

    def locate_information(self):
        # Job: Use RAG retrieval to locate potential pages containing the user_input information
        # Returns: List of page numbers (sorted) where the information may be, based on top-k similarity
        if not self.index:
            raise ValueError("Vectorstore not built.")

        k = 3  # Retrieve top 3 pages; adjust as needed
        query = self.user_input  # Use user_input as the query for semantic search
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search FAISS index
        similarities, indices = self.index.search(query_embedding, k)
        
        # Get page numbers from top matches (filtering out low similarity if needed)
        page_numbers = sorted([self.documents[idx]['page'] for idx in indices[0] if similarities[0][indices[0].tolist().index(idx)] > 0.1])  # Threshold to avoid irrelevant pages
        
        return page_numbers if page_numbers else None  # Return None if no relevant pages found
    
    def _get_base64_images(self, page_numbers):
        image_b64 = {}

    def analyze_page(self, page_numbers: list):
        # takes a screen shot of the the pages given and tries to locate information that the user wants 
        # returns generated response. 
        
        

        return 
    
    def analyze_dataframes_dict(self, pages):
        # works in parallel with analyze_page
        # will look over the set of pages in dataframes dict 
        return 
    
    def gather_information_and_report(self, analyze_page_outputs, analyze_dataframes_dict_output):
        # gathers the info from both analyze_page and analyze_dataframes_dict functions and generates findings
        return 
    
    async def main(self):
        return 
    

