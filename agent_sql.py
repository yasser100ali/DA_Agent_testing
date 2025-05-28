import utils
from agents import Agent
import json
import streamlit as st
import sqlite3
import pandas as pd
import time


DATABASE_FILE = "testing_data/chinook.db"

def table_to_dataframe(table_name, db_file=DATABASE_FILE):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        query = f"SELECT * FROM {table_name}".upper()
        df = pd.read_sql_query(query, conn)
        return df
    
    except Exception as e:
        st.write(f"Error has occured: {str(e)}")
        return None
    
    finally:
        if conn:
            conn.close()

def list_tables(db_file):
    """
    Lists all tables in the specified SQLite database file.

    Args:
        db_file (str): The path to the SQLite database file.

    Returns:
        list: A list of table names, or None if an error occurs.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Execute a query to fetch the names of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        # The result is a list of tuples, so we extract the first element of each tuple
        return [table[0].upper() for table in tables]
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if conn:
            conn.close()

def query_to_dataframe(db_file, query, params=None):
    """
    Executes a custom SELECT SQL query and returns the results as a pandas DataFrame.

    Args:
        db_file (str): The path to the SQLite database file.
        query (str): The SELECT SQL query to execute.
        params (tuple, optional): Parameters to substitute into the query for safety. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing the query results, or None if an error occurs.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        print(f"An SQLite error occurred in fetch_query_to_dataframe: {e}")
        return None
    except Exception as e: # Catch pandas related errors too
        print(f"A general error occurred in fetch_query_to_dataframe: {e}")
        return None
    finally:
        if conn:
            conn.close()

def list_table_features(db_file, table_name):
    """
    Lists all features (columns) and their data types for a given table.

    Args:
        db_file (str): The path to the SQLite database file.
        table_name (str): The name of the table.

    Returns:
        list: A list of tuples, where each tuple contains (column_name, data_type),
              or None if an error occurs or the table doesn't exist.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # PRAGMA table_info is a special command to get table structure
        # It's generally safe from SQL injection for table_name if table_name is validated
        # or comes from a trusted source (like output of list_tables).
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns_info = cursor.fetchall()
        if not columns_info:
            print(f"Table '{table_name}' not found or has no columns.")
            return None
        # Each row from PRAGMA table_info contains:
        # (cid, name, type, notnull, dflt_value, pk)
        # We are interested in name (index 1) and type (index 2)
        return [info[1] for info in columns_info]
    except sqlite3.Error as e:
        print(f"An error occurred with list_table_features for table '{table_name}': {e}")
        return None
    except Exception as e:
        print(f"A general error occurred in list_table_features: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_table_features(db_file):
    """
    Retrieves all tables in the database and their respective features (columns and types).

    Args:
        db_file (str): The path to the SQLite database file.

    Returns:
        dict: A dictionary where keys are table names (str) and values are lists
              of tuples, each tuple containing (column_name, data_type).
              Returns None if listing tables fails, or an empty dict if no tables exist.
              Individual tables might have a value of None if fetching their features failed.
    """
    all_tables = list_tables(db_file)
    if all_tables is None:
        print("Failed to retrieve list of tables.")
        return None

    if not all_tables:
        print("No tables found in the database.")
        return {}

    database_features_dict = {}
    for table_name in all_tables:
        features = list_table_features(db_file, table_name)
        # list_table_features returns a list of (name, type) tuples,
        # or None on error, or an empty list if table has no columns/not found by PRAGMA.
        database_features_dict[table_name] = features
        if features is None:
            print(f"Could not retrieve features for table '{table_name}'. It will have a None value in the dictionary.")
        elif not features: # Empty list
             print(f"Table '{table_name}' has no features listed or could not be found by PRAGMA table_info.")


    return database_features_dict

class sqlAgent:
    def __init__(self, user_input):
        self.user_input = user_input
        self.functions = {
            "table_to_dataframe": table_to_dataframe,
            "query_to_dataframe": query_to_dataframe,
            "get_all_table_features": get_all_table_features
        }
        self.available_tables = list_tables(DATABASE_FILE)
  
    def get_table_name(self):
        system_prompt = """
        You are part of a SQL retrieval agent. 
        Your job is very simple, simply take the query and extract the table name from it and return as json.
        If there are multiple tables then simply return them all. 
        Store the tables or table in a key called "tables" as shown below and that should lead to an array of the table(s). 

        Example: "From the (nameofdatabase) extract coaches players and games"

        outputs:
        There are 3 tables in this request, "coaches", "players" and "games"

        ```json
        {
            "tables": ["COACHES", "PLAYERS", "GAMES"]
        }
        ```

        Example 2: "From the sql database, extract the records table."
        
        outputs: 
        There is one table to extract from here. 
        ```json
        {
            "tables": ["RECORDS"]
        }
        """
        user_input = self.user_input 
        user_input += f". \nHere are the list of tables from the database: {self.available_tables}. If user misspells the table-name, correct it according to this list"
        json_output = Agent(user_input, self.functions).json_agent(system_prompt, "model", is_visible=False)

        table_names_from_llm = json_output[list(json_output.keys())[0]]

        current_prompt_id = st.session_state.get('prompt_id', 0) # Default to 0 if not set


        loaded_tables_count = 0
        loaded_table_names = []

        for table_name_query in table_names_from_llm:
            # Find the closest match from self.available_tables to handle case differences from LLM
            # This step is crucial if the LLM doesn't perfectly match the case from your DB

            actual_table_name = None
            for db_table in self.available_tables:
                if db_table.upper() == table_name_query.upper():
                    actual_table_name = db_table
                    break
            
            if not actual_table_name:
                msg = f"SQL Agent: Table '{table_name_query}' suggested by LLM not found in the database table list: {self.available_tables}."
                st.warning(msg)
                st.session_state.messages[current_prompt_id].append({"role": "assistant", "content": msg})
                continue

            df = table_to_dataframe(actual_table_name) 
            df = utils.standardize_file(df)
        
            if df is not None:
                st.session_state.dataframes_dict[actual_table_name.upper()] = df
                loaded_tables_count += 1
                loaded_table_names.append(actual_table_name)
            else:
                error_msg = f"SQL Agent: Failed to load table `{actual_table_name}`."
                # table_to_dataframe should have already shown an st.error
                st.session_state.messages[current_prompt_id].append({"role": "assistant", "content": error_msg})

        str_tables = str(table_names_from_llm)

        system_prompt = f"""
        You are an agent part of a larger system. 

        Simply response something along the lines "Ok I have imported the tables, look to your right... no sorry my right, your left" have a little humor like this but change this up.
        or sometimes just a simple "Ok I've imported the tables (table names). Anything else?" 
        
        So just relay the tables are imported and sprinkle a little humor every now and then. 
    
        The table names will be provided by user prompt. 

        Also write out the table names in bold and normal grammer. So if a table name is "PLAYER_STATS" write it as **Player Stats**. 
        """

        user_input = self.user_input

        user_input += f"\nHere are the list of tables that you have imported: {str_tables}"
        chat_response = Agent(user_input, {}).chat(system_prompt, "")
        utils.assistant_message('chat', chat_response)

        #time.sleep(30)
    
    def main(self):
        self.get_table_name()
