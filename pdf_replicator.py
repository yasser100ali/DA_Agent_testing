import fitz 
import pandas as pd


def replicate_pdf(pdf_file):
    """
    Processes a PDF file by extracting text and tables from each page using PyMuPDF.
    Focuses on text and tables only; includes post-processing for cleaner tables.
    Adjusted table detection to use 'text' strategy for better capture of borderless tables.
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    results = []

    def clean_table_data(raw_data):
        """
        Post-processes raw table extract to handle merged cells, empties, and narratives.
        Added handling for concatenated cell content by splitting on newlines and re-parsing.
        """
        if not raw_data or len(raw_data) < 2:
            return {"cleaned_table": [], "note": "No substantial data in table."}
        
        # First, handle cases where data is concatenated in cells
        processed_data = []
        for row in raw_data:
            new_rows = [row]  # Default to original row
            if any(isinstance(cell, str) and '\n' in cell for cell in row if cell is not None):
                # Split long cells
                max_lines = max(cell.count('\n') + 1 for cell in row if isinstance(cell, str))
                split_rows = [[] for _ in range(max_lines)]
                for cell in row:
                    if isinstance(cell, str):
                        lines = cell.split('\n')
                        for i, line in enumerate(lines):
                            split_rows[i].append(line.strip())
                        for i in range(len(lines), max_lines):
                            split_rows[i].append('')
                    else:
                        for sr in split_rows:
                            sr.append(cell)
                new_rows = [sr for sr in split_rows if any(sr)]  # Skip empty
            processed_data.extend(new_rows)
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Forward-fill for merged cells
        df = df.ffill(axis=0).ffill(axis=1)
        
        # Drop fully empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
        
        # Filter narrative rows (e.g., bullets starting with '•' or long text)
        df = df[~df.apply(lambda row: row.astype(str).str.contains('•|driven by|offset by|variance', regex=True).any(), axis=1)]
        
        # Infer and set headers if possible (check for keywords like 'Actuals', 'Budget')
        if len(df) > 0:
            potential_header_row = df.iloc[0].astype(str)
            if any(word in ' '.join(potential_header_row.values) for word in ['Actual', 'Budget', 'Variance', 'Actuals']):
                df.columns = potential_header_row
                df = df[1:].reset_index(drop=True)
            else:
                # Fallback: Use generic column names if no clear header
                df.columns = [f"Col_{i}" for i in range(len(df.columns))]
        
        # Additional cleanup: Strip newlines and extra spaces
        df = df.apply(lambda col: col.astype(str).str.replace('\n', ' ').str.replace(r'\s+', ' ', regex=True).str.strip() if col.dtype == 'object' else col)
        
        return {"cleaned_table": df.to_dict(orient="records"), "note": f"Cleaned {len(df)} rows."}

    for i, page in enumerate(doc, start=1):
        # Extract structured text as dict (blocks with positions)
        text_blocks = page.get_text("dict")["blocks"]
        extracted_text = []
        for block in text_blocks:
            if "lines" in block:
                block_text = " ".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
                if block_text:
                    extracted_text.append({
                        "text": block_text,
                        "bbox": block["bbox"]
                    })
        
        # Find and extract tables with adjusted parameters for better detection
        tables = page.find_tables(strategy="text", snap_tolerance=3)  # Switched to "text" for borderless tables, reduced tolerance
        extracted_tables = []
        for tab in tables:
            raw_data = tab.extract()  # Raw list of lists
            
            # Post-process
            cleaned = clean_table_data(raw_data)
            
            extracted_tables.append({
                "raw_data": raw_data,  # Keep for debugging
                **cleaned,
                "bbox": tab.bbox
            })
        
        # Filter text to exclude table regions (avoid duplication)
        filtered_text = [tb for tb in extracted_text if not any(
            # Check if text bbox overlaps significantly with table bbox
            (tb["bbox"][0] < t["bbox"][2] and tb["bbox"][2] > t["bbox"][0] and
             tb["bbox"][1] < t["bbox"][3] and tb["bbox"][3] > t["bbox"][1])
            for t in extracted_tables
        )]
        
        # Append results for this page
        results.append({
            "page": i,
            "text": filtered_text if filtered_text else [{"text": "No non-table text extracted."}],
            "tables": extracted_tables
        })
    
    doc.close()  # Clean up
    return results

