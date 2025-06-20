import pandas as pd
import numpy as np
import streamlit as st


def create_llm_summary_view(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a "summary view" DataFrame to be used as input for an LLM.

    This function returns a new DataFrame of the same shape where:
    1. Each cell's content is annotated with its pandas .iloc coordinate, 
       e.g., "Value (row, col)".
    2. Long rows of numbers are condensed to show only the first 2 and last 2.

    Args:
        df_raw (pd.DataFrame): The unprocessed DataFrame from the Excel sheet.

    Returns:
        pd.DataFrame: A new DataFrame containing the annotated and condensed summary.
    """
    # Create a new DataFrame of the same size to store our formatted strings
    # Initializing with empty strings and object dtype for mixed content
    summary_df = pd.DataFrame(index=df_raw.index, columns=df_raw.columns, dtype=object).fillna('')

    # Iterate through each row to process it
    for r_idx, row in df_raw.iterrows():
        # Find all numeric values in the row to decide if we need to condense
        numeric_values = [v for v in row if isinstance(v, (int, float, np.number))]
        
        kept_numeric_values = set()
        should_condense = len(numeric_values) > 4

        if should_condense:
            # If we need to condense, identify the first 2 and last 2 numbers to keep
            kept_numeric_values.update(numeric_values[:2])
            kept_numeric_values.update(numeric_values[-2:])
        
        ellipsis_placed = False
        for c_idx, cell_value in enumerate(row):
            # Skip completely empty/null cells
            if pd.isna(cell_value):
                continue
            
            # Get the integer-based coordinates for the label
            coord_str = f"({r_idx}, {c_idx})"
            
            # --- Format the cell content ---
            formatted_content = ""
            if isinstance(cell_value, (int, float, np.number)):
                # For numbers, check if they should be kept or replaced by "..."
                if not should_condense or cell_value in kept_numeric_values:
                    formatted_content = f"{cell_value} {coord_str}"
                elif not ellipsis_placed:
                    formatted_content = "..."
                    ellipsis_placed = True
            else:
                # For non-numeric types, just format as string
                cell_text = str(cell_value).strip()
                if cell_text: # Don't process empty strings
                    formatted_content = f"{cell_text} {coord_str}"
            
            # Assign the formatted string to the corresponding cell in our summary DataFrame
            summary_df.iloc[r_idx, c_idx] = formatted_content
            
    return summary_df

# --- EXAMPLE USAGE ---
# if __name__ == '__main__':
#     # Create a dummy DataFrame that mimics your Excel sheet's structure
#     data = {
#         0: ["", "2025 Budget", "Calendar Days", "Provider Admissions", "ED Vists"],
#         1: [None, None, 31, 4574, 12.3],
#         2: [None, None, 28, 4432, 12.3],
#         3: [None, None, 31, 4594, 12.3],
#         4: [None, None, 30, 4273, 12.3],
#         5: [None, None, 31, 4245, 12.3],
#         6: [None, None, 30, 4193, 12.3],
#         7: [None, None, 31, 4073, 12.3],
#         8: [None, None, 31, 4721, 12.3]
#     }
#     df_raw = pd.DataFrame(data).T # Transpose to get the right shape
    
#     # Generate the summary view DataFrame for the LLM
#     summary_for_llm =

st.title("Testing ")

excel_sheet = pd.read_excel("testing_data/DOE_data.xlsx", sheet_name=None)
st.write(excel_sheet["SSF_DOE"])

st.write(create_llm_summary_view(excel_sheet["SSF_DOE"]))

st.write(excel_sheet["SSF_DOE"].iloc[7,4])