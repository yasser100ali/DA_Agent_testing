from agents import Agent
import pandas as pd
import time

def name_unnamed_features(df):
    feature_names = list(df.columns)
    unnamed_features = {}

    for index, feature in enumerate(feature_names):
        if "UNNAMED" in feature.upper():
            unnamed_features[feature.upper()] = {
                "feature_name": feature.upper(),
                "index": index,
                "column_contents": df[feature_names[index]] 
            }

    system_prompt = """
    You are a feature naming agent. Your job is to take a list of features that are currently unnamed, 
    look over their contents (provided as a string sample) and give a new name. 
    The new name MUST be a single word or words connected by underscores, ALL CAPITALIZED.
    
    You are given a function "update_column_name(df, column_index, new_name)" which you MUST use.
    'df' is already available in your execution scope.

    Your output MUST be a Python script defining a single function main() that calls 
    update_column_name for each feature you rename and then returns the modified df.

    Example of your output if you decide to rename column at index 14 to "MONTH" and column at index 18 to "EMPLOYEES":
    ```python
    def main():
        # Renaming feature at index 14 to "MONTH"
        update_column_name(df=df, column_index=14, new_name="MONTH")
        # Renaming feature at index 18 to "EMPLOYEES"
        update_column_name(df=df, column_index=18, new_name="EMPLOYEES")
        return df
    ```
    Only provide the Python code block. Do not add any other explanations.
    """


    def update_column_name(df, column_index, new_name):
        current_columns = list(df.columns)
        current_columns[column_index] = new_name
        df.columns = current_columns
        

    local_var = {
        "df": df, 
        "update_column_name": update_column_name
    }

    user_input = f"""
    Here are the list of features to update. Look over the contents and come up with an appropriate and simple name in all caps. 

    {unnamed_features}
    """

    agent = Agent(user_input, local_var)
    
    updated_df, code = agent.coder(system_prompt, hide_code=True)

    return updated_df

