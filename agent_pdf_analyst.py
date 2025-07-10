from openai import AsyncOpenAI
import asyncio
import fitz  # PyMuPDF
import streamlit as st
import base64
import os
import json
import hashlib
from typing import Any, List, Dict

# --- Caching Configuration ---
CACHE_DIR = "pdf_replications"

class PDFAgent:
    """
    An agent to process PDF pages concurrently while respecting API rate limits.
    """
    def __init__(self):
        # It's best practice to load the API key from st.secrets or an environment variable
        self.client = AsyncOpenAI(api_key="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")

    async def _get_response_for_image(self, image_b64: str, page_num: int) -> str:
        """
        Sends a single page image to the OpenAI API for data replication.
        """
        system_prompt = "You are an expert data entry assistant. Replicate the data in the image and organize the data accordingly into a pandas dataframe where the features are in the top row. Output the python code for the dataframe in a markdown block."
        user_prompt_payload = [
            {"type": "text", "text": f"Replicate the content from this image into a structured dataframe that is organized for page {page_num}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"}}
        ]

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_payload}
            ]

            response = await self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                max_tokens=4000
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"An error occurred while processing page {page_num}: {e}"

    async def replicate_pdf(self, pdf_file: Any) -> List[Dict[str, Any]]:
        """
        Processes a PDF file by creating throttled, concurrent tasks for each page.
        """
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        semaphore = asyncio.Semaphore(5)
        tasks = []

        async def process_page_with_semaphore(page_num: int, image_b64: str) -> str:
            async with semaphore:
                return await self._get_response_for_image(image_b64, page_num)

        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            task = asyncio.create_task(process_page_with_semaphore(i, img_b64))
            tasks.append(task)
        
        replications = await asyncio.gather(*tasks)

        return [
            {"page": i, "replication_output": content}
            for i, content in enumerate(replications, start=1)
        ]
    

def get_file_hash(file: Any) -> str:
    """Calculates the SHA256 hash of a file's content."""
    file.seek(0) # Ensure we're reading from the start
    file_bytes = file.read()
    file.seek(0) # Reset cursor for other functions
    return hashlib.sha256(file_bytes).hexdigest()

def save_results_to_cache(file_hash: str, results: List[Dict[str, Any]]):
    """Saves replication results to a JSON file in the cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    with open(cache_path, 'w') as f:
        json.dump(results, f, indent=4)

def load_results_from_cache(file_hash: str) -> List[Dict[str, Any]] | None:
    """Loads results from a cache file if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None