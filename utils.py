# General functions that will be used throughout.

import streamlit as st 
from openai import OpenAI
import re
import json
import traceback
import logging
import copy
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import pandas as pd
import uuid
import numpy as np
import time
import os
from datetime import datetime, timezone
import base64

is_local_hosting=False

def get_response(system_prompt, user_prompt, model, show_stream=True, local_hosting=is_local_hosting):
    # when running the models locally
    if local_hosting:
        print('\n\nHere is the system prompt.\n\n')
        print(system_prompt)
        print('\n\nHere is the user prompt.\n\n')
        print(user_prompt)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        # model = "hf.co/unsloth/Qwen3-32B-GGUF:Q8_K_XL"

        # response = ollama.chat(
        #     model=model,
        #     messages=messages,
        #     stream=show_stream
        # )
    
        # return response
        
    # when not running locally, either due to no root access or some other issue
    else:
        client = OpenAI(
            api_key="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"
        )
    
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True  # Enable streaming
        )
    
        return response

def display_stream(response, visible=True):

    # Create a Streamlit container to hold the streaming content
    content = ""
    if visible is True:
        container = st.empty()
        
    # Iterate through the streaming response
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            # Append the new token to the full content
            chunk = chunk.choices[0].delta.content
            chunk = chunk.replace('$', '\$')
            content += chunk

            # Update the container with the current content
            if visible is True:
                container.write(content)
                
    return content
        
def typewriter_effect(text, delay = 0.027):
    words = text.split() 
    for word in words:
        yield word + " "
        time.sleep(delay) 


def extract_json(input_str: str):
    """
    Extracts the last complete JSON object found within curly braces {} 
    from an input string. Uses a greedy regex match for the initial block.
    Attempts to fix common issues like trailing commas, single quotes for strings/keys,
    and Python-style booleans/None. It will also attempt to parse only the first
    JSON object if trailing non-JSON data is present after fixes.
    """
    # Pattern to find a block enclosed in curly braces.
    # The original function used a greedy match to find the last such block.
    pattern = r'(\{.*\})' 
    matches = re.findall(pattern, input_str, re.DOTALL)
    
    if not matches:
        # Handle cases where no {}-enclosed block was found at all
        return "Pattern '{}' not found in the input string."

    # Use the last match found (as per the original logic)
    match_content = matches[-1].strip()

    # Step 1: Get rid of trailing commas before a closing brace or bracket.
    clean_match = re.sub(r',\s*([}\]])', r'\1', match_content)
    
    try:
        # Attempt to parse the initially cleaned JSON string directly
        json_dict = json.loads(clean_match)
        return json_dict
        
    except json.JSONDecodeError as e_initial:
        # If direct parsing fails, attempt to fix common LLM-generated issues.
        try:
            # Start with the comma-cleaned string
            fixed_str = clean_match
            
            # Step 2: Replace Pythonic None, True, False with JSON null, true, false.
            # This is done *before* quote changes for unquoted keywords.
            fixed_str = re.sub(r'\bNone\b', 'null', fixed_str)
            fixed_str = re.sub(r'\bTrue\b', 'true', fixed_str)
            fixed_str = re.sub(r'\bFalse\b', 'false', fixed_str)
            
            # Step 3: Convert single-quoted strings/keys to double-quoted strings/keys.
            fixed_str = re.sub(r"'((?:\\.|[^'])*)'", r'"\1"', fixed_str)
            
            # Step 4: Attempt to parse the fixed string using raw_decode.
            # This will parse the first valid JSON object and allow for ignoring
            # subsequent non-JSON text (like LLM "thinking" text).
            decoder = json.JSONDecoder()
            json_dict, _ = decoder.raw_decode(fixed_str) # We don't strictly need the end_index here
            return json_dict # Successfully extracted the JSON object
            
        except json.JSONDecodeError as e_fixed_final:
            # This exception means that fixed_str is malformed even for raw_decode.
            # This could be due to issues within the JSON structure itself, not just trailing data.
            error_context_length = 75 

            # Details from the initial parsing attempt (e_initial on clean_match)
            original_error_pos = e_initial.pos
            original_doc = e_initial.doc 
            original_start = max(0, original_error_pos - error_context_length)
            original_end = min(len(original_doc), original_error_pos + error_context_length)
            original_snippet = original_doc[original_start:original_end]
            original_relative_pos = original_error_pos - original_start
            original_marked_snippet = original_snippet[:original_relative_pos] + "[ERROR->" + original_snippet[original_relative_pos:]

            # Details from the final parsing attempt (e_fixed_final on fixed_str)
            fixed_error_pos = e_fixed_final.pos
            fixed_doc = e_fixed_final.doc 
            fixed_start = max(0, fixed_error_pos - error_context_length)
            fixed_end = min(len(fixed_doc), fixed_error_pos + error_context_length)
            fixed_snippet = fixed_doc[fixed_start:fixed_end]
            fixed_relative_pos = fixed_error_pos - fixed_start
            fixed_marked_snippet = fixed_snippet[:fixed_relative_pos] + "[ERROR->" + fixed_snippet[fixed_relative_pos:]

            return (f"Initial JSON parsing failed: {e_initial.msg} at position {e_initial.pos}. "
                    f"Problematic text (original): ...{original_marked_snippet}...\n"
                    f"Attempted fixes (quotes, keywords), but parsing the fixed string also failed: {e_fixed_final.msg} at position {e_fixed_final.pos}. "
                    f"Problematic text (after fixes): ...{fixed_marked_snippet}...\n"
                    f"Original string (after trailing comma removal) snippet: {clean_match[:250]}...\n"
                    f"String after attempted fixes snippet: {fixed_str[:250]}...")

def extract_python(input_str):
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, input_str, re.DOTALL)

    return matches[0]

def convert_to_features_list(dataframes_dict: dict) -> dict:
    """
    Recursively traverses the dataframes_dict and creates a nested dictionary
    representing the full schema of files, sheets, sections, and columns.

    This function can handle:
    1. Simple CSVs (file -> df)
    2. Excel files with structured sheets (file -> sheet -> df)
    3. Excel files with unstructured sheets that have been sectioned off
       (file -> sheet -> section -> df)
    """
    features_schema = {}
    
    # Level 1: Iterate through files
    for filename, file_content in dataframes_dict.items():
        
        # Case 1: The file is a simple CSV DataFrame
        if isinstance(file_content, pd.DataFrame):
            features_schema[filename] = list(file_content.columns)
            
        # Case 2: The file is an Excel file (a dictionary of sheets)
        elif isinstance(file_content, dict):
            features_schema[filename] = {}
            
            # Level 2: Iterate through sheets in the Excel file
            for pagename, page_content in file_content.items():
                
                # Sub-case 2a: The sheet is a single structured DataFrame
                if isinstance(page_content, pd.DataFrame):
                    features_schema[filename][pagename] = list(page_content.columns)
                
                # Sub-case 2b: The sheet has been broken into sections (a dict of DataFrames)
                elif isinstance(page_content, dict):
                    features_schema[filename][pagename] = {}
                    
                    # Level 3: Iterate through sections in the sheet
                    for section_name, section_df in page_content.items():
                        if isinstance(section_df, pd.DataFrame):
                            features_schema[filename][pagename][section_name] = list(section_df.columns)
                            
    return features_schema

def execute_code(code, local_var): 
    logger = logging.getLogger('error_logger')
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler('error_log_data_analyst_agent.txt')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)

    try:
        namespace = copy.deepcopy(local_var)
        exec(code, namespace)
        result = namespace['main']()
        success = True
        
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        error_message = "Traceback:\n"
        for frame in tb:
            filename, line_number, function_name, text = frame
            error_message += f"File: {filename}, Line: {line_number}, in {function_name}\n"
            error_message += f"  {text}\n"
        error_message += f"Error: {e}"

        # logging error
        logger.error(error_message)                    
        print('\n\nError Message\n\n')
        print(error_message)
        result = error_message
        success = False
        
    return result, success
    
def show_output(result):
    """
    Display different result types in Streamlit with error handling.
    """
    try:
        if isinstance(result, tuple):
            st.write('Recursing this bitch')
            result = list(result)
            show_output(result)
        
        # Case 1: result is a list
        if isinstance(result, list):

            for item in result:
                try:
                    if isinstance(item, go.Figure):
                        st.plotly_chart(item, key=str(uuid.uuid4()))
                    elif isinstance(item, pd.DataFrame):
                        st.dataframe(item, key=str(uuid.uuid4()))
                    elif isinstance(item, dict):
                        show_output(item)
                    elif isinstance(item, tuple):
                        item = list(item)
                        show_output(item)
                    else:
                        st.write(item)

                except Exception as e:
                    st.error(f"Error displaying list item: {e}")
        
        # Case 2: result is a dictionary
        elif isinstance(result, dict):
            for key, item in result.items():

                try:
                    if isinstance(item, go.Figure):
                        st.plotly_chart(item, key=str(uuid.uuid4()))
                    elif isinstance(item, pd.DataFrame):
                        st.dataframe(item, key=str(uuid.uuid4()))
                    else:
                        st.write(item)

                except Exception as e:
                    st.error(f"Error displaying dictionary item for key '{key}': {e}")
        
        # Case 3: result is a Plotly figure
        elif isinstance(result, go.Figure):
            try:
                st.plotly_chart(result, key=str(uuid.uuid4()))

            except Exception as e:
                st.error(f"Error displaying Plotly figure: {e}")
        
        # Case 4: result is a DataFrame
        elif isinstance(result, pd.DataFrame):
            try:
                st.dataframe(result, key=str(uuid.uuid4()))

            except Exception as e:
                st.error(f"Error displaying DataFrame: {e}")
        
        # Catch-all for anything else
        else:
            try:
                st.write(result)

            except Exception as e:
                st.error(f"Error displaying result: {e}")
                
    except Exception as e:
        st.error(f"Unexpected error in _show_result: {e}")


def filter_figures(item):
    if isinstance(item, Figure) or \
        (isinstance(item, str) and item.strip().startswith("Figure(")):
        return None 
    
    elif isinstance(item, list):
        new_list = []
        for sub_item in item:
            processed_sub_item = filter_figures(sub_item)
            if processed_sub_item is not None:
                new_list.append(processed_sub_item)
        return new_list 

  
    else:
        if isinstance(item, str):
            return item.strip()
        else:
            return item


def assistant_message(item_type, item):
    prompt_id = st.session_state.get('prompt_id')

    if prompt_id is None:
        st.error("Error in assistant_message: prompt_id not found in session_state.")
        # Potentially log this or handle more gracefully depending on application structure
        return

    # Ensure the messages list for the prompt_id exists
    st.session_state.setdefault('messages', {}) # Ensure 'messages' key exists
    st.session_state['messages'].setdefault(prompt_id, []) # Ensure prompt_id list exists

    messages_for_prompt = st.session_state['messages'][prompt_id]

    # Scenario 1: This is the first part of the assistant's response for this prompt.
    # This typically means only the user's message exists so far.
    if len(messages_for_prompt) == 0 or (len(messages_for_prompt) == 1 and messages_for_prompt[0]['role'] == 'user'):
        new_assistant_message = {
            'role': 'assistant',
            'content': {
                item_type: item
            }
        }
        # If only user message exists, append. If no messages, this is the first.
        if len(messages_for_prompt) == 1 :
             messages_for_prompt.append(new_assistant_message)
        else: # len is 0, very unusual but defensive. Add as the first message.
            messages_for_prompt.append(new_assistant_message)
            st.warning(f"Warning: Assistant message added for prompt_id {prompt_id} before any user message. This might indicate an issue in call order.")


    # Scenario 2: An assistant message structure already exists (at index 1, after user message at index 0).
    # Add the new item_type to its 'content' dictionary, or update if None, or add uniquely if already filled.
    elif len(messages_for_prompt) > 1 and messages_for_prompt[1]['role'] == 'assistant':
        assistant_message_obj = messages_for_prompt[1]

        # Ensure 'content' key exists and is a dictionary
        if not isinstance(assistant_message_obj.get('content'), dict):
            st.warning(f"Warning: Assistant message 'content' for prompt_id {prompt_id} was not a dict. Re-initializing.")
            assistant_message_obj['content'] = {}

        assistant_content_dict = assistant_message_obj['content']

        # Check if the item_type key already exists
        if item_type not in assistant_content_dict:
            # Key doesn't exist, so add it.
            assistant_content_dict[item_type] = item
        elif assistant_content_dict[item_type] is None:
            # Key exists and its value is None, so update it.
            assistant_content_dict[item_type] = item
        else:
            # Key exists and is already populated with a non-None value.
            # Create a new unique key for this new item to avoid overwriting.
            unique_item_type = f"{item_type}_{str(uuid.uuid4())[:8]}" # Using a shorter UUID for readability
            assistant_content_dict[unique_item_type] = item
    else:
        # This state is unexpected (e.g., len > 1 but index 1 is not an assistant message, or len is 0 but previous block didn't catch)
        st.error(f"Error in assistant_message: Unexpected message structure for prompt_id {prompt_id}. Length: {len(messages_for_prompt)}")
        # Fallback: try to append as a new assistant message, but this indicates a potential logic flaw elsewhere.
        # This might happen if the assumption about user message at index 0 and assistant at index 1 is violated.
        fallback_message = {
            'role': 'assistant',
            'content': {
                item_type: item
            }
        }
        messages_for_prompt.append(fallback_message)
    
    

def standardize_file(df_input, default_year=2025, **kwargs):
    """
    Standardizes a DataFrame by:
    1. Uppercasing and replacing spaces with underscores in column names.
    2. Identifying a potential date/time column.
    3. Converting and standardizing this date/time column to 'YYYY-MM-DD' format,
       naming it 'DATE', and moving it to the front.
       - If a 'MONTH' column (with month names) is found, it uses this with a 'YEAR'
         column (if available) or default_year, defaulting to the 1st day of the month.
       - Otherwise, it attempts to parse the identified time column using pd.to_datetime
         and formats valid dates to 'YYYY-MM-DD'.
    4. Standardizing other string column values (uppercase, strip, replace space with underscore).

    Args:
        df_input (pd.DataFrame): The input DataFrame.
        default_year (int, optional): The default year to use if a year column is not found
                                      or contains missing values when processing a 'MONTH' column.
                                      Defaults to 2025.
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        pd.DataFrame: The standardized DataFrame.
                      Returns an empty DataFrame if the input is empty.

    Raises:
        TypeError: If df_input is not a pandas DataFrame.
    """

    POSSIBLE_TIME_COLUMNS = [
        'DATE', 'MONTH', 'TIME', 'PERIOD', 'YEARMONTH', 'DATETIME', 'TIMESTAMP',
        'DATE_TIME', 'DT', 'TRANS_DT', 'EVENT_DATE', 'ACTIVITY_DATE'
    ]
    MONTH_MAP = {
        'JAN': 1, 'JANUARY': 1, 'FEB': 2, 'FEBRUARY': 2, 'MAR': 3, 'MARCH': 3,
        'APR': 4, 'APRIL': 4, 'MAY': 5, 'JUN': 6, 'JUNE': 6, 'JUL': 7, 'JULY': 7,
        'AUG': 8, 'AUGUST': 8, 'SEP': 9, 'SEPT': 9, 'SEPTEMBER': 9, 'OCT': 10, 'OCTOBER': 10,
        'NOV': 11, 'NOVEMBER': 11, 'DEC': 12, 'DECEMBER': 12
    }

    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df_input.empty:
        return pd.DataFrame() # Return empty DataFrame if input is empty

    df = df_input.copy() # Work on a copy

    # --- 1. Standardize Column Names ---
    try:
        df.columns = [str(col).upper().strip().replace(' ', '_') for col in df.columns]
        standardized_cols = df.columns
    except Exception as e:
        # print(f"Warning: Could not standardize column names: {e}") # Optional logging
        standardized_cols = df.columns # Use original names if standardization fails

    # --- 2. Identify and Process Potential Date Column ---
    time_col_name = next((col for col in POSSIBLE_TIME_COLUMNS if col in standardized_cols), None)
    processed_date_col_name = None # Track if date processing was successful
    new_date_col = 'DATE'          # Target name for the processed date column

    if time_col_name:
        try:
            original_dtype = df[time_col_name].dtype
            temp_standardized_dates_str = pd.Series(index=df.index, dtype=str) # Store 'YYYY-MM-DD' strings

            is_likely_month_col = False
            if time_col_name == 'MONTH' and pd.api.types.is_string_dtype(original_dtype):
                try:
                    unique_vals = df[time_col_name].dropna().astype(str).str.upper().unique()
                    if any(m in MONTH_MAP for m in unique_vals[:10]):
                        is_likely_month_col = True
                except Exception:
                    pass # Ignore errors during this heuristic check

            if is_likely_month_col:
                # --- Handle Month Name Column Case (e.g., 'MONTH' column with 'JAN', 'FEB') ---
                month_nums = df[time_col_name].astype(str).str.upper().map(MONTH_MAP)
                base_valid_idx = month_nums.notna() # Rows where month name was successfully mapped

                # Determine the year to use for each row
                year_values_for_construction = pd.Series(index=df.index, dtype='object')
                actual_default_year = default_year if default_year is not None else pd.Timestamp.now().year

                if 'YEAR' in standardized_cols:
                    year_series_numeric = pd.to_numeric(df['YEAR'], errors='coerce')
                    year_values_for_construction = year_series_numeric.fillna(actual_default_year)
                else:
                    year_values_for_construction.fillna(actual_default_year, inplace=True)

                # Ensure both month and year are valid for construction
                # and year is a whole number (int)
                final_valid_idx = base_valid_idx & \
                                  year_values_for_construction.notna() & \
                                  (year_values_for_construction.apply(lambda x: isinstance(x, (int, float)) and x == int(x)))


                if final_valid_idx.any():
                    years_str = year_values_for_construction[final_valid_idx].astype(int).astype(str)
                    months_str = month_nums[final_valid_idx].astype(int).astype(str).str.zfill(2)
                    day_str = "01" # Default to the 1st day of the month

                    temp_standardized_dates_str.loc[final_valid_idx] = years_str + '-' + months_str + '-' + day_str
            
            else:
                # --- Handle Other Potential Date/Time Column Cases ---
                # Attempt standard datetime conversion
                # Pandas to_datetime can infer many formats. errors='coerce' turns unparseable dates into NaT.
                datetime_col = pd.to_datetime(df[time_col_name], errors='coerce')
                valid_idx = datetime_col.notna() # Find where conversion succeeded
                
                if valid_idx.any():
                    # Format valid datetime objects to 'YYYY-MM-DD' string
                    temp_standardized_dates_str.loc[valid_idx] = datetime_col[valid_idx].dt.strftime('%Y-%m-%d')

            # --- Assign results to DataFrame and clean up ---
            if not temp_standardized_dates_str.isnull().all():
                df[new_date_col] = temp_standardized_dates_str.replace({np.nan: None})
                processed_date_col_name = new_date_col

                if time_col_name != new_date_col and time_col_name in df.columns:
                    df = df.drop(columns=[time_col_name])
                
                # Move the new/processed 'DATE' column to the front
                cols = [processed_date_col_name] + [col for col in df.columns if col != processed_date_col_name]
                df = df[cols]
            else:
                if new_date_col in df.columns and df[new_date_col].isnull().all():
                    df = df.drop(columns=[new_date_col], errors='ignore')

        except Exception as e:
            # print(f"Warning: Could not process time column '{time_col_name}': {e}") # Optional logging
            processed_date_col_name = None # Ensure it's marked as failed

    # --- 3. Standardize String Column Values ---
    try:
        string_cols = df.select_dtypes(include=['object', 'string']).columns

        if processed_date_col_name and processed_date_col_name in string_cols:
            # Check dtype just in case it was converted to something else unexpectedly
            # The 'DATE' column should now contain strings like 'YYYY-MM-DD' or None
            if pd.api.types.is_object_dtype(df[processed_date_col_name].dtype) or \
               pd.api.types.is_string_dtype(df[processed_date_col_name].dtype):
                string_cols = string_cols.drop(processed_date_col_name)

        for col in string_cols:
            if col in df.columns: # Check column still exists
                # Ensure we only apply string methods to actual string data, handling NaNs
                # Convert to string, then apply operations. NaNs become 'nan' string.
                # Strip whitespace, uppercase, replace space with underscore.
                # df.loc[:, col] = df[col].astype(str).str.strip().str.upper().str.replace(' ', '_')
                
                # More robust handling for mixed types / NaNs before string conversion
                mask_notna = df[col].notna()
                df.loc[mask_notna, col] = df.loc[mask_notna, col].astype(str).str.strip().str.upper().str.replace(' ', '_')


    except Exception as e:
        # print(f"Warning: Could not standardize string column values: {e}") # Optional logging
        pass

    # --- 4. Return Standardized DataFrame ---
    return df

def convert_features_list_to_array(features_list):
    collected_words_set = set()

    def extract_all_strings(element):
        """
        Recursively extracts all strings from a nested structure,
        capitalizes them, and adds them to collected_words_set.
        """
        if isinstance(element, str):
            collected_words_set.add(element.upper())
        elif isinstance(element, list):
            for item in element:
                extract_all_strings(item) # Recurse for each item in the list
        elif isinstance(element, dict):
            for key, value in element.items():
                collected_words_set.add(key.upper()) # Add dictionary key
                extract_all_strings(value)      # Recurse for dictionary value

    # Start the extraction process with your main data structure
    extract_all_strings(features_list)

    # Convert the set to a list
    result_array = list(collected_words_set)
    return result_array


def auto_correct_code_by_list(
    code_string: str,
    correct_words_list: list,
    only_correct_quoted: bool = True  # Default changed to True as it's often for quoted strings
) -> str:
    """
    Auto-corrects words in a code string based on a list of correct words.
    Specifically enhanced to handle stripping of common extensions if the base
    word matches an entry in correct_words_list.

    Args:
        code_string: The input string containing the code.
        correct_words_list: A list of correctly spelled words/phrases (canonical keys).
        only_correct_quoted: If True, only attempts to correct tokens that
                             appear to be directly enclosed in single ('') or
                             double ("") quotes.
    Returns:
        The code string with auto-corrections applied.
    """

    lower_to_canonical_map = {word.lower(): word for word in correct_words_list}
    correct_words_lowercase_set = set(lower_to_canonical_map.keys())
    common_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt'] # Add more if needed

    def get_corrected_token_internal(token_str_input):
        """Applies correction logic to a single token."""
        token_lower = token_str_input.lower()

        # Attempt 1: Direct match (existing logic for case/length)
        if token_lower in correct_words_lowercase_set:
            canonical_word = lower_to_canonical_map[token_lower]
            if len(token_str_input) == len(canonical_word):
                return canonical_word

        # Attempt 2: Check if removing a common DataFrame extension makes it a match.
        # This is particularly for correcting things like 'INVOICES.csv' to 'INVOICES'
        # if 'INVOICES' is in correct_words_list.
        for ext in common_extensions:
            if token_lower.endswith(ext):
                token_without_ext = token_str_input[:-len(ext)]
                token_without_ext_lower = token_lower[:-len(ext)]
                if token_without_ext_lower in correct_words_lowercase_set:
                    canonical_word_for_base = lower_to_canonical_map[token_without_ext_lower]
                    # Ensure the canonical word IS the base name and lengths match for the base part
                    if len(token_without_ext) == len(canonical_word_for_base): # and token_without_ext_lower == canonical_word_for_base.lower() (redundant due to map key)
                        return canonical_word_for_base # Return the corrected base name, stripping the extension
                # If an extension was found and processed (match or not),
                # we typically don't want to then treat the dot in the extension as a part separator.
                # So, if a token ended with a known extension, we stop this token's processing here.
                # If it was corrected, great. If not, it means the base name wasn't in correct_words_list.
                return token_str_input # Return original token if base after stripping ext didn't match

        # Attempt 3: Part-based correction (existing logic, for non-extension related issues)
        # This will only be reached if Attempt 1 and 2 (extension stripping) did not return.
        primary_separator = None
        if "_" in token_str_input:
            primary_separator = "_"
        elif "." in token_str_input: # Now, this dot is less likely to be a recognized extension
            primary_separator = "."

        if primary_separator:
            parts = token_str_input.split(primary_separator)
            corrected_parts = []
            made_change_in_parts = False

            for i, part_str in enumerate(parts):
                # Handle empty strings from splits like "A__B" or "A." correctly
                if not part_str:
                    if i < len(parts) - 1 or (i == len(parts) -1 and token_str_input.endswith(primary_separator)): # "A__B" or "A."
                        corrected_parts.append("")
                        # If it's an empty part not due to trailing separator, it might need a change flag if it *was* something
                    else: # Should not happen if split produces empty string not at end unless original was just separator
                         corrected_parts.append(part_str) # Should be empty if it's a true end part
                    continue


                corrected_part_val = part_str  # Default to original part
                part_lower = part_str.lower()

                if part_lower in correct_words_lowercase_set:
                    canonical_part = lower_to_canonical_map[part_lower]
                    if len(part_str) == len(canonical_part):
                        corrected_part_val = canonical_part

                if corrected_part_val != part_str:
                    made_change_in_parts = True
                corrected_parts.append(corrected_part_val)

            if made_change_in_parts:
                return primary_separator.join(corrected_parts)

        # If no correction applied, return the original token
        return token_str_input

    # Regex to find "tokens" (sequences of alphanumeric characters, underscores, and dots).
    # It also captures surrounding quotes if only_correct_quoted is True.
    if only_correct_quoted:
        # This regex captures (quote)(token_inside_quotes)(quote)
        # or (non_whitespace_non_quote_characters)
        # We are interested in correcting the token_inside_quotes.
        # This needs careful handling to replace only the inside part.
        # A simpler approach for quoted strings: find all string literals first.

        # Simpler strategy for quoted strings:
        # Find all string literals, correct their content, then reconstruct.
        def correct_quoted_content(match_obj):
            quote_char = match_obj.group(1) # The quote character (' or ")
            string_content = match_obj.group(2) # The content inside the quotes
            corrected_content = get_corrected_token_internal(string_content)
            return f"{quote_char}{corrected_content}{quote_char}"

        # Regex for '...' or "..."
        # It handles escaped quotes inside the string to some extent but might not be perfect for all edge cases.
        code_string = re.sub(r"""(["'])((?:\\.|(?!\1).)*?)\1""", correct_quoted_content, code_string)
        return code_string # Return after quote-specific correction

    # Fallback to general token processing if not only_correct_quoted
    # (This part would need careful review if used, as correcting non-quoted tokens
    # like variable names broadly can be risky. The quoted correction is safer here.)
    else: # if not only_correct_quoted
        token_pattern = re.compile(r"([a-zA-Z0-9_.]+)") # Original token pattern
        last_end_index = 0
        result_parts = []
        for match in token_pattern.finditer(code_string):
            start_index_token, end_index_token = match.span()
            original_token = match.group(1)

            result_parts.append(code_string[last_end_index:start_index_token])
            corrected_token = get_corrected_token_internal(original_token)
            result_parts.append(corrected_token)
            last_end_index = end_index_token

        result_parts.append(code_string[last_end_index:])
        return "".join(result_parts)
    
def is_subset_dictionary(subset_dict, main_dict):

    """
    Checks if the first dictionary (subset_dict) is a subset of the second dictionary (main_dict),
    considering nested structures for file and sheet/column organization.

    Args:
        subset_dict (dict): The dictionary to check if it's a subset.
        main_dict (dict): The dictionary to check against.

    Returns:
        bool: True if subset_dict is a subset of main_dict, False otherwise.
               Prints error messages indicating where the subset condition fails.
    """

    for file_name, file_content_subset in subset_dict.items():
        # Check 1: File name exists in the main dictionary
        if file_name not in main_dict:
            print(f"Error: File '{file_name}' from the subset is not found in the main dictionary.")
            return False

        file_content_main = main_dict[file_name]

        # Check 2: File content types match (list for CSV-like, dict for Excel-like)
        if type(file_content_subset) != type(file_content_main):
            print(f"Error: Structure mismatch for file '{file_name}'. "
                f"Subset has type {type(file_content_subset).__name__} while main has type {type(file_content_main).__name__}.")
            return False

        # Case A: File content is a list of columns (e.g., CSV)
        if isinstance(file_content_subset, list):
            for column in file_content_subset:
                if column not in file_content_main:
                    print(f"Error: Column '{column}' in file '{file_name}' from the subset "
                        f"is not found in the main dictionary's corresponding file.")
                    return False

        # Case B: File content is a dictionary of sheets (e.g., XLSX)
        elif isinstance(file_content_subset, dict):
            for sheet_name, columns_subset in file_content_subset.items():
                # Check B.1: Sheet name exists in the main dictionary's file content
                if sheet_name not in file_content_main:
                    print(f"Error: Sheet '{sheet_name}' in file '{file_name}' from the subset "
                        f"is not found in the main dictionary's corresponding file.")
                    return False

                columns_main = file_content_main[sheet_name]

                # Check B.2: Sheet content types match (must be list)
                if not isinstance(columns_subset, list) or not isinstance(columns_main, list):
                    print(f"Error: Structure mismatch for sheet '{sheet_name}' in file '{file_name}'. "
                        f"Both subset and main dictionary sheet contents should be lists of columns.")
                    if not isinstance(columns_subset, list):
                        print(f"       Subset sheet '{sheet_name}' content is type {type(columns_subset).__name__}.")
                    if not isinstance(columns_main, list):
                        print(f"       Main dictionary sheet '{sheet_name}' content is type {type(columns_main).__name__}.")
                    return False

                # Check B.3: All columns in the subset sheet exist in the main dictionary's sheet
                for column in columns_subset:
                    if column not in columns_main:
                        print(f"Error: Column '{column}' in sheet '{sheet_name}' of file '{file_name}' from the subset "
                            f"is not found in the main dictionary's corresponding sheet.")
                        return False
        else:
            # This case should ideally not be reached if types are pre-validated or known
            print(f"Error: Unsupported content type for '{file_name}' in the subset dictionary: {type(file_content_subset).__name__}.")
            return False
            
    return True


def get_current_time_components():
    now = datetime.now()
    month_name = now.strftime("%m")
    week_number = now.isocalendar()[1]
    date_day_key = now.strftime("%m_%d_%Y_%A")
    return month_name, week_number, date_day_key


def is_dataframe_structured(df, nan_threshold=0.6, header_nan_threshold=0.7, generic_col_threshold=0.5):
    """
    Determines if a DataFrame is structured based on heuristics.
    Args:
        df (pd.DataFrame): The DataFrame to check.
        nan_threshold (float): If sparsity (NaN ratio) > this, considered unstructured.
        header_nan_threshold (float): If first/second data rows' NaN ratio > this, considered unstructured.
        generic_col_threshold (float): If ratio of 'Unnamed:' columns > this, considered unstructured.
    Returns:
        bool: True if structured, False otherwise.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False # Empty or non-DataFrame is considered not structured for this purpose

    # 1. Check overall sparsity
    total_cells = df.size
    nan_cells = df.isnull().sum().sum()
    sparsity = nan_cells / total_cells if total_cells > 0 else 0
    if sparsity > nan_threshold:
        # print(f"DEBUG: Unstructured by overall sparsity: {sparsity:.2f} > {nan_threshold}")
        return False

    # 2. Check for generic column names (e.g., "Unnamed: X")
    if df.columns.empty:
        # print("DEBUG: Unstructured by empty columns")
        return False

    unnamed_cols = sum(1 for col in df.columns if isinstance(col, str) and col.startswith("Unnamed:"))
    if len(df.columns) > 0 and (unnamed_cols / len(df.columns)) > generic_col_threshold:
        # If many columns are unnamed, check the first data row.
        if not df.empty:
            first_row_nan_ratio = df.iloc[0].isnull().sum() / len(df.columns) if len(df.columns) > 0 else 0
            if first_row_nan_ratio > header_nan_threshold:
                # print(f"DEBUG: Unstructured by generic columns ({unnamed_cols}/{len(df.columns)}) and sparse first row ({first_row_nan_ratio:.2f})")
                return False

    # 3. Check the first few data rows for sparsity
    if not df.empty:
        first_data_row_sparsity = df.iloc[0].isnull().sum() / len(df.columns) if len(df.columns) > 0 else 0
        if first_data_row_sparsity > header_nan_threshold:
            if len(df) > 1:
                second_data_row_sparsity = df.iloc[1].isnull().sum() / len(df.columns) if len(df.columns) > 0 else 0
                if second_data_row_sparsity > header_nan_threshold:
                    # print(f"DEBUG: Unstructured by sparse first two data rows: R1({first_data_row_sparsity:.2f}), R2({second_data_row_sparsity:.2f})")
                    return False
            else: # Only one row and it's very sparse
                # print(f"DEBUG: Unstructured by single sparse data row: R1({first_data_row_sparsity:.2f})")
                return False
    
    # print("DEBUG: Classified as Structured")
    return True



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


        # --- 4. Generate metadata for the new file ---
        # The data can be a single DataFrame (CSV) or a dict of DataFrames (Excel)
        is_excel = isinstance(data_in_session, dict)
        sheets_to_process = data_in_session if is_excel else { "primary_sheet": data_in_session }

        file_level_meta = {
            # Note: Since we don't have the raw file, we hash the DataFrame content.
            # This is still very effective for detecting data changes.
            "content_hash": int(pd.util.hash_pandas_object(sheets_to_process[next(iter(sheets_to_process))], index=True).sum()),
            "estimated_size_bytes": int(sum(df.memory_usage(deep=True).sum() for df in sheets_to_process.values())),
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
        


    # --- 5. Save the updated dictionary back to the JSON file ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(file_details, f, indent=4)

   