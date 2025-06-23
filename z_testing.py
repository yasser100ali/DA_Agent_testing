import streamlit as st
import utils 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re 



st.title("Structuring data with images of data")

def load_data(uploaded_file_obj):
    """
    Loads data from an uploaded file.
    Returns a pandas DataFrame for CSVs or a dictionary of DataFrames for Excel files.
    """
    try:
        uploaded_file_obj.seek(0)
        file_name = uploaded_file_obj.name.lower()
        if file_name.endswith('.csv'):
            return pd.read_csv(uploaded_file_obj)
        elif file_name.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
            # Pass the file object directly, openpyxl (used by read_excel) can handle it
            return pd.read_excel(uploaded_file_obj, sheet_name=None, engine='openpyxl')
        else:
            st.error(f"Unsupported file type: {uploaded_file_obj.name}")
            return None
    except Exception as e:
        st.error(f'Error reading file {uploaded_file_obj.name}: {e}')
        return None


def convert_to_float_if_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Goes column by column through a DataFrame and converts the
    data type to float if the entire column is numeric.

    Args:
        df: The input pandas DataFrame.

    Returns:
        The DataFrame with numeric columns converted to float type.
    """
    for col in df.columns:
        # Attempt to convert the column to a numeric type.
        # 'errors=coerce' will turn any non-numeric values into NaN (Not a Number).
        numeric_series = pd.to_numeric(df[col], errors='coerce')

        # If the conversion doesn't result in all values being NaN,
        # it means the column was numeric. Then we can change the type.
        if not numeric_series.isnull().all():
            df[col] = numeric_series.astype(float)
    return df


def upload_and_format_files():
    """
    Handles file uploads, standardizes data, capitalizes file and sheet names, 
    and processes unstructured sheets using a cache to avoid re-computation.
    """
    uploaded_files = st.file_uploader(
        'Upload Data Files',
        type=['csv', 'xlsx', 'xls', 'xlsm', 'xlsb'],
        accept_multiple_files=True,
        key="file_uploader_capitalized_integrated"
    )
    
    dataframes_dict = {}
    
    # <<< MODIFIED >>> This will now collect equations from all processed files
    all_equations = {}

    is_structured = True

    if uploaded_files:
        for uploaded_file_obj in uploaded_files:
            original_file_name = uploaded_file_obj.name
            capitalized_file_name = ""
            try:
                uploaded_file_obj.seek(0)
                base_name, extension = os.path.splitext(original_file_name)
                capitalized_file_name = base_name.upper() + extension.upper()

                loaded_data = load_data(uploaded_file_obj) 

                if loaded_data is None:
                    continue

                standardized_data_output = None

                if isinstance(loaded_data, dict): # Excel file
                    capitalized_standardized_sheets = {}
                    for original_sheet_name, df_sheet_original in loaded_data.items():
                        capitalized_sheet_name = original_sheet_name.upper()
                        df_for_processing = df_sheet_original

                        if isinstance(df_for_processing, pd.DataFrame):
                            is_structured = utils.is_dataframe_structured(df_for_processing)
                            
                            if not is_structured:
                                st.write("This bad boy needs some structure!")
                            # Standardize the chosen DataFrame (either original, cached, or newly processed)
                            capitalized_standardized_sheets[capitalized_sheet_name] = utils.standardize_file(df_for_processing)
                        else:
                            capitalized_standardized_sheets[capitalized_sheet_name] = df_for_processing
                            st.warning(f"Sheet '{capitalized_sheet_name}' in '{capitalized_file_name}' is not a DataFrame.")
                    
                    standardized_data_output = capitalized_standardized_sheets
                
                elif isinstance(loaded_data, pd.DataFrame): # CSV file
                    standardized_data_output = utils.standardize_file(loaded_data)
                else:
                    st.warning(f"Unexpected data type after loading {capitalized_file_name}: {type(loaded_data)}")
                    continue

                if standardized_data_output is not None:
                    dataframes_dict[capitalized_file_name] = standardized_data_output
            except Exception as e:
                st.error(f"Failed to process file {capitalized_file_name or original_file_name}: {e}")

    if not is_structured:
        return dataframes_dict
    else:
        return dataframes_dict
    
def create_compact_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a highly condensed and annotated "skeleton" view of a DataFrame for an LLM.

    This function aggressively compacts the data by:
    1. Ignoring all completely empty rows.
    2. For each row, gathering only the non-empty cells.
    3. If a row has more than 4 non-empty cells, it keeps only the first 2
       and last 2, placing '...' in between.
    4. Annotating each kept cell with its original (row, col) coordinate.
    """
    all_processed_rows = []

    # Iterate through each row of the original raw DataFrame
    for r_idx, row in df_raw.iterrows():
        # 1. Gather only the non-empty cells from the row, keeping their original coordinates
        content_items = []
        for c_idx, cell_value in enumerate(row):
            if pd.notna(cell_value) and str(cell_value).strip() != '':
                content_items.append({'value': cell_value, 'r': r_idx, 'c': c_idx})

        # If the row was effectively empty, skip it entirely
        if not content_items:
            continue

        # 2. Condense the list of content items if it's too long
        condensed_items = []
        if len(content_items) > 4:
            # Keep first two, add ellipsis, keep last two
            condensed_items.extend(content_items[:2])
            condensed_items.append({'value': '...', 'r': None, 'c': None}) # Ellipsis marker
            condensed_items.extend(content_items[-2:])
        else:
            # If the row is short, keep all its content
            condensed_items = content_items

        # 3. Format each item for the final output row
        formatted_row = []
        for item in condensed_items:
            if item['value'] == '...':
                formatted_row.append('...')
            else:
                # Annotate with original coordinates
                formatted_row.append(f"{item['value']} ({item['r']}, {item['c']})")
        
        all_processed_rows.append(formatted_row)

    # 4. Create a final, compact DataFrame from the processed rows
    # Pandas will automatically handle the jagged rows by filling with None/NaN
    summary_df = pd.DataFrame(all_processed_rows)

    # Set clean, generic column headers
    summary_df.columns = [f'Col_{i}' for i in range(summary_df.shape[1])]

    return summary_df


def slice_dataframe_by_sections(df_raw: pd.DataFrame, section_titles: list) -> dict:
    """
    Slices a raw DataFrame into a dictionary of cleaned, separate DataFrames
    by performing a full search for each section title to find its exact location.

    This version is based on the robust "search all cells" strategy.

    Args:
        df_raw (pd.DataFrame): The original, untouched DataFrame from the Excel sheet.
        section_titles (list): A list of strings corresponding to the section titles.

    Returns:
        dict: A dictionary where keys are snake_cased section titles and
              values are the corresponding cleaned pandas DataFrames.
    """
    final_dataframes = {}
    
    if df_raw.empty:
        return {}

    # --- 1. Search every cell to find the exact (row, col) of each title ---
    title_locations = {}
    title_set = set(section_titles)
    
    print("--- Searching for section titles in the DataFrame ---")
    for r_idx in range(len(df_raw)):
        for c_idx in range(len(df_raw.columns)):
            cell_value = str(df_raw.iloc[r_idx, c_idx]).strip()
            if cell_value in title_set:
                # Store the title and its precise (row, col) coordinate
                title_locations[cell_value] = (r_idx, c_idx)
                print(f"Found title '{cell_value}' at location ({r_idx}, {c_idx})")
                # Remove from set to avoid finding the same title twice (optional optimization)
                title_set.remove(cell_value)

    if not title_locations:
        print("ERROR: None of the provided section titles were found anywhere in the DataFrame.")
        return {}
        
    print("--- Title search complete ---")

    # 2. Sort the found titles by their row number to process them in order
    sorted_titles = sorted(title_locations.items(), key=lambda item: item[1][0])

    # 3. Iterate through the sorted titles to define boundaries and slice
    for i, (title, (start_row, start_col)) in enumerate(sorted_titles):
        
        # Determine the end row for the current section's slice
        if i + 1 < len(sorted_titles):
            end_row = sorted_titles[i + 1][1][0] - 1 # End before the next title starts
        else:
            end_row = len(df_raw) - 1 # It's the last section, go to the end
            
        print(f"Slicing section '{title}' from row {start_row} to {end_row}...")

        # 4. Slice the original DataFrame to get the section block
        section_df = df_raw.iloc[start_row:end_row + 1].copy()

        # 5. Clean and organize the extracted block
        section_df.dropna(how='all', axis=0, inplace=True)
        section_df.dropna(how='all', axis=1, inplace=True)
        section_df.reset_index(drop=True, inplace=True)
        
        # Heuristic to find and promote the best header row within the slice
        if not section_df.empty:
            header_row_index = 0 # Default to the first row (the title row)
            min_nulls = float('inf')
            # Look for a row with the least number of empty cells to be the header
            for idx, row in section_df.iterrows():
                # Don't consider the title row itself as a potential data header if other options exist
                if idx > 0 and row.isnull().sum() < min_nulls:
                    min_nulls = row.isnull().sum()
                    header_row_index = idx
            
            # Set the new headers and drop that row from the data
            if len(section_df) > header_row_index:
                 section_df.columns = section_df.iloc[header_row_index]
                 section_df = section_df.drop(header_row_index).reset_index(drop=True)

        # 6. Create a clean key and add the final DataFrame to our dictionary
        key_name = re.sub(r'[^A-Z0-9_]+', '', title.replace(' ', '_').upper())
        final_dataframes[key_name] = section_df
        
    return final_dataframes


def convert_df_to_image(df, output_path="dataframe_screenshots/temp_sheet_image.png"):
    """
    Converts a pandas DataFrame to a PNG image file using Matplotlib.
    This is a robust alternative to dataframe-to-image.
    """
    if df is None or df.empty:
        print("Cannot convert an empty DataFrame to an image.")
        return False
        
    # Create a figure and axes for the plot
    # The figure size will depend on the dataframe size. You may need to adjust figsize.
    fig, ax = plt.subplots(figsize=(15, df.shape[0] * 0.5)) # Adjust size as needed
    
    # Hide the axes
    ax.axis('off')
    ax.axis('tight')
    
    # Create the table from the dataframe
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='left'
    )
    
    # Adjust table properties if needed
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Save the figure to the specified path
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig) # Close the figure to free up memory
        print(f"✅ DataFrame successfully exported to {output_path} using Matplotlib.")
        return True
    except Exception as e:
        print(f"Error converting DataFrame to image with Matplotlib: {e}")
        plt.close(fig)
        return False

dataframes_dict = upload_and_format_files()

if dataframes_dict:
    # for file, file_dict in dataframes_dict.items():
    #     for sheet_name, df in file_dict.items():
    #         st.write(sheet_name)
    #         st.write(df)
    #         summary_df = create_compact_summary(df)
    #         convert_df_to_image(summary_df, output_path=f"dataframe_screenshots/{summary_df.size}")

    st.write(dataframes_dict)

    file_dict = dataframes_dict["DOE_DATA.XLSX"]
    df_ssf = file_dict["SSF_DOE"]

    st.write(df_ssf)
    ai_identified_titles = [
        "2025_BUDGET",
        "2025_FORECAST",
        "EXPENSE_ASSUMPTIONS",
        "EXPENSES_FORECAST",
        "VARIANCE_TO_BUDGET"    
    ]

    structured_data = slice_dataframe_by_sections(df_ssf, ai_identified_titles)

    for section, df in structured_data.items():
        st.write(section)
        st.markdown(df.to_markdown())

