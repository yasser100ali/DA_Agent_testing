from agents.agents import ReAct
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader  # Assuming PyPDF2 is installed; if not, pip install PyPDF2
import base64 
import streamlit as st 
from pdf2image import convert_from_bytes
import io 
from openai import OpenAI, AsyncOpenAI
import utils.utils as utils
from agents.agents import Agent

class PDFDeepInsights:
    def __init__(self, user_input, local_var):
        self.user_input = user_input
        self.local_var = local_var
        # Initialize RAG components once in init for efficiency
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.documents = []
        self.index = None
        self.pdf_stream = None
        self._prepare_pdf_stream()
        self._build_vectorstore()
        self.client = OpenAI(api_key="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")

    def _prepare_pdf_stream(self):
        pdf_input = self.local_var["pdf_file_objects"]
        pdf_bytes = None

        if isinstance(pdf_input, dict):
            if pdf_bytes is None:
                # Search values for file-like or bytes (handles {filename: UploadedFile} structure)
                for key, value in pdf_input.items():
                    if isinstance(value, bytes):
                        pdf_bytes = value
                        break
                    elif hasattr(value, 'read') and hasattr(value, 'seek'):
                        value.seek(0)
                        pdf_bytes = value.read()
                        break
                    elif hasattr(value, 'getvalue'):
                        pdf_bytes = value.getvalue()
                        break
                if pdf_bytes is None:
                    raise ValueError(f"Cannot extract PDF bytes from dict with keys: {list(pdf_input.keys())}. No bytes or file-like found in values. Check structure.")
        elif isinstance(pdf_input, str):  # file path
            with open(pdf_input, 'rb') as f:
                pdf_bytes = f.read()
        elif hasattr(pdf_input, 'read') and hasattr(pdf_input, 'seek'):  # file-like object, e.g., UploadedFile
            pdf_input.seek(0)
            pdf_bytes = pdf_input.read()
        elif isinstance(pdf_input, bytes):
            pdf_bytes = pdf_input
        else:
            raise TypeError(f"Unsupported type for pdf_file_objects: {type(pdf_input)}")

        self.pdf_stream = io.BytesIO(pdf_bytes)


    def _build_vectorstore(self):
        if self.pdf_stream is None:
            raise ValueError("PDF stream not prepared.")
        self.pdf_stream.seek(0)
        reader = PdfReader(self.pdf_stream)
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
        relevant_indices = [idx for i, idx in enumerate(indices[0]) if similarities[0][i] > 0.1]
        page_numbers = sorted([self.documents[idx]['page'] for idx in relevant_indices])
        
        return page_numbers if page_numbers else None  # Return None if no relevant pages found

    def temp_locate_information(self) -> list:
        # a temporary version of locate_information that simply uses a basic llm call to find the page numbers. 
        # dependent on the user to provide where the information probably is 

        agent = Agent(self.user_input)
        system_prompt = """
        Take a look over the user prompt, which should contain a page number of multiple page numbers.
        Return the page numbers as a list in a json style dict. 

        example: 
        user_prompt: "Go through pages 5, 12, and 15 and get the following information... (rest is not relevant, only need the page numbers)"

        output:
        ```json
        {
            "page_numbers": [5, 12, 15]
        }
        ```
        """

        json_output = agent.json_agent(system_prompt=system_prompt, user_input=self.user_input)

        # outputs as a list 
        return json_output["page_numbers"]

    def _get_base64_images(self, page_numbers):
        image_b64 = {}

        if self.pdf_stream is None or not page_numbers:
            return image_b64

        self.pdf_stream.seek(0)

        # converts the specified pages to images
        sorted_pages = sorted(page_numbers)

        images = []
        for page_num in sorted_pages:
            page_image = convert_from_bytes(self.pdf_stream.getvalue(), first_page=page_num, last_page=page_num, dpi=600)
            if page_image:
                images.append(page_image[0])

        print(f"\nPage Numbers: {sorted_pages}\nLength of images: {len(images)}")

        for i, image in enumerate(images):
            page_num = sorted_pages[i]
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            b64_image = base64.b64encode(buffered.getvalue()).decode()
            image_b64[page_num] = b64_image
        
        return image_b64

    def analyze_page(self, image_b64: dict):
        # takes the dictionary of images and imports them into an LLM, which analyzes the image and generates a response based on the user prompt and image content
        system_prompt = "Analyze the image given to you and answer the user's question accordingly. If the answer is not on the given page, simply say so."
        user_content = [{"type": "text", "text": self.user_input}]

        for base64_str in image_b64.values():
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_str}"}
            })
        
        response = utils.get_response(system_prompt=system_prompt, user_prompt=user_content, show_stream=True, model="gpt-4o")
        llm_response = utils.display_stream(response)

        utils.assistant_message("chat", llm_response)

        return llm_response
    
    def analyze_dataframes_dict(self, pages):
        # works in parallel with analyze_page
        # will look over the set of pages in dataframes dict 
        return 
    
    def gather_information_and_report(self, analyze_page_outputs, analyze_dataframes_dict_output):
        # gathers the info from both analyze_page and analyze_dataframes_dict functions and generates findings
        return 
    
    def main(self):
        # version 1 is meant to be non parallel and simple
        # the main objective is to extract the page numbers (which we'll have to better refine) and import those page images into an LLM which will then answer the user's prompt

        page_numbers = self.temp_locate_information()
        image_b64_dict = self._get_base64_images(page_numbers=page_numbers)
        llm_response = self.analyze_page(image_b64=image_b64_dict)

        return llm_response   

