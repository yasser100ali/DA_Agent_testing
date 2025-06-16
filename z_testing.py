import pandas as pd
import json
import os
from datetime import datetime, timezone
import streamlit as st

def sync_file_metadata_from_session():
    """
    Loads 'file_details.json', compares it with st.session_state.dataframes_dict,
    and generates metadata for any new files found in the session state.
    """
    json_path = "message_history/file_details.json"

    # --- 1. Exit if there's nothing to process ---
    if 'dataframes_dict' not in st.session_state or not st.session_state.dataframes_dict:
        print("Session state is empty. No metadata to sync.")
        return

    # --- 2. Load existing details or initialize a new dictionary ---
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "r", encoding="utf-8") as f:
            file_details = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        file_details = {}
        print("Existing file_details.json not found or empty. Starting a new one.")

    # --- 3. Iterate through files currently in the session state ---
    for filename, data_in_session in st.session_state.dataframes_dict.items():
        if filename in file_details:
            # If the file already has metadata, skip it.
            continue

        print(f"⏳ New file '{filename}' found in session. Generating metadata...")

        # --- 4. Generate metadata for the new file ---
        # The data can be a single DataFrame (CSV) or a dict of DataFrames (Excel)
        is_excel = isinstance(data_in_session, dict)
        sheets_to_process = data_in_session if is_excel else { "primary_sheet": data_in_session }

        file_level_meta = {
            # Note: Since we don't have the raw file, we hash the DataFrame content.
            # This is still very effective for detecting data changes.
            "content_hash": pd.util.hash_pandas_object(sheets_to_process[next(iter(sheets_to_process))], index=True).sum(),
            "estimated_size_bytes": sum(df.memory_usage(deep=True).sum() for df in sheets_to_process.values()),
            "last_updated_timestamp": datetime.now(timezone.utc).isoformat(),
            "sheets": {}
        }

        for sheet_name, df in sheets_to_process.items():
            # DataFrame/Sheet-Level Metadata
            sheet_level_meta = {
                "row_count": int(df.shape[0]),
                "column_count": int(df.shape[1]),
                "duplicate_row_count": int(df.duplicated().sum()),
                "dataframe_summary": "",  # To be filled by an AI agent later
                "columns": {}
            }

            # Column-Level Metadata
            for col_name in df.columns:
                col = df[col_name]
                col_meta = {
                    "type": str(col.dtype),
                    "missing_value_count": int(col.isnull().sum()),
                    "missing_value_percentage": float(col.isnull().sum() / len(col) if len(col) > 0 else 0),
                    "unique_value_count": int(col.nunique()),
                    "is_potential_id": col.nunique() == len(col),
                    "description": ""
                }

                if pd.api.types.is_numeric_dtype(col):
                    stats = col.describe().to_dict()
                    col_meta["statistics"] = {k: float(v) for k, v in stats.items()}
                else:
                    unique_cats = col.dropna().unique()
                    col_meta["categories"] = unique_cats.tolist()[:100]

                sheet_level_meta["columns"][str(col_name)] = col_meta

            file_level_meta["sheets"][sheet_name] = sheet_level_meta

        # Add the fully generated entry to our main dictionary
        file_details[filename] = file_level_meta
        print(f"✅ Generated metadata for '{filename}'.")


    # --- 5. Save the updated dictionary back to the JSON file ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(file_details, f, indent=4)

    print("✔️ Metadata sync complete.")
