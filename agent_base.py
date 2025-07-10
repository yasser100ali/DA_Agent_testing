import utils
from agents import Agent
import streamlit as st
import pandas as pd
import time 


class BaseAgent:
    def __init__(self, user_input, local_var={}):
        self.user_input = user_input
        self.local_var = local_var
        self.dataframes_dict = local_var['dataframes_dict']
        self.agent = Agent(user_input, local_var)

    def base(self):
        orchestrator_prompt = """
            You are an orchestrator agent. 

            Take the user prompt and assign it to a subagent, "agent". Here are you subagents:
                a. visualization
                    = for when a prompt is asking to create some kind of visualization 
                b. analysis
                    - For when rather basic Exploratory Data Analysis is called for using pandas
                c. machine_learning
                    - when predictive modeling using scikit-learn libary may be used           
                    
            Follow this format: 
             "Create a scatter chart of Avg wait from ED and assign to depart delay." 
    
            Output: 
            Since the user is asking to create a visualization, this task will be delegated to the visulization_agent
            ```json
            {
                "agent": "visualization",
            }

            Your only item here should be "agent". 
            ```
        """
        
        agent = Agent(self.user_input, self.local_var)
        json = agent.json_agent(orchestrator_prompt, self.model)

        st.write("**Filtering Data**")
        features_list = str(utils.convert_to_features_list(self.dataframes_dict)) 
        features_list_json = utils.extract_json(features_list)
        
        word_count = len(features_list.split())
        
        if word_count > 150: 
            # this is the case where we use an agent to filer out the data to ease the coder agent's attention
            input_features_list = filter_data(self.user_input, features_list)
            input_features_list_json = utils.extract_json(input_features_list)
           
        else:
            # amount of information from data is sufficiently small that adding a pre-processing data filtration step is unneccessary 
            input_features_list = features_list
            input_features_list_json = features_list_json
            
        st.session_state['features_list_json'] = input_features_list_json

        is_subset = utils.is_subset_dictionary(input_features_list_json, features_list_json)
    
        if is_subset:
            st.write('This is a subset')
            
            standard =  f"""
            1. Write complete self-contained code that includes all necessary imports
            2. You are given 'dataframes_dict' which contains all the files, pages (pages only applies to non-csv excel files), and feature names. Make sure to spell according to the parenthesis input for these details. To get access to a particular page, you would type 'dataframes_dict['excelsheetname.xlsx']['page'] or if it is a csv dataframes_dict['file.csv']
            3. Define a `main` function that encapsulates the task. The `main` function should:
            - Perform the requested task.
            - Return the result
            4. Ensure the code is syntactically correct and ready to execute. Use markdown in your formatting: ```python import ... def main(): ... ```


            ### Whenver you get a feature, make sure you verify that feature is actually in the following list: {input_features_list}
            **IF** attempting to merge dataframes, always inspect the key columns you intend to merge on (e.g., 'Date', 'Month'). Check their data types (dtype) and examine sample values to ensure they are compatible. The features_list provides column names but not their format or type.

            It is also very important to note that **ALL FEATURES AND CATEGORIES ARE CAPTILIZED**. So capitalize the user's features and categories. 
            """
        
            system_prompts = {
                'visualization': standard + """                    
                You are visualization_agent.
                Your job is to create visualizations using **plotly** (ONLY plotly NOT matplotlib). 
                Use pandas to get relevant information (if needed), and plotly to graph. 



                example: 
                input_prompt: 
                "create a bar graph of predicted mean grouped by the categories in line of business"
                output: 
                ```python
                import pandas as pd
                import plotly.express as px
                
                def main():
                    # Retrieve the dataframe from the provided dictionary
                    df = dataframes_dict['membership_predictions_0221.csv']
                
                    # Group by LINE_OF_BUSINESS and calculate the mean of predicted_mean
                    grouped_df = df.groupby('LINE_OF_BUSINESS')['predicted_mean'].mean().reset_index()
                
                    # Create the bar chart using Plotly Express
                    output = px.bar(
                        grouped_df,
                        x='LINE_OF_BUSINESS',
                        y='predicted_mean',
                        title='Average Predicted Membership by Line of Business',
                        labels={'predicted_mean': 'Average Predicted Mean', 'LINE_OF_BUSINESS': 'Line of Business'},
                        text='predicted_mean'  # Display values on bars
                    )
                
                    # Improve layout and formatting
                    output.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                    output.update_layout(
                        xaxis_tickangle=-45,  # Rotate x-axis labels for readability
                        uniformtext_minsize=8,
                        uniformtext_mode='hide'
                    )
                
                    return output
                ```
                
                input_prompt: 
                "show revenue for Antioch and mantica commercial over the course of the year. Have this be a bar graph."
                output: 
                ```python
                import pandas as pd
                import plotly.express as px

        def main():
            # Load the relevant dataframe
            df = dataframes_dict['tabular_data.xlsx']['tabular_data']

            # Find the top 5 payers by frequency
            top_payers = df['Primary Payer'].value_counts().nlargest(5).index
            df_top_payers = df[df['Primary Payer'].isin(top_payers)]

            # Create an interactive bar chart with plotly
            fig = px.bar(
                df_top_payers,
                x='Primary Payer',
                y='Bed TAT',
                title='Bed Turnaround Time for Top 5 Payers',
                labels={'Bed TAT': 'Bed Turnaround Time (minutes)', 'Primary Payer': 'Primary Payer'}
            )
            
            # for figs do not convert to json, leave as fig. 
            return fig

        example 3: 

        input_prompt:
        "Use a RandomForest model to find the top 5 most important features that influence 'Avg wait from ED'."            
                
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        import json

        def main():
            # Load the dataframe
            df = dataframes_dict['tabular_data.xlsx']['tabular_data']

            # Prepare data for modeling
            df_clean = df.dropna(subset=['Avg wait from ED'])
            df_clean = df_clean.select_dtypes(include=['number']) # Use only numeric columns
            
            X = df_clean.drop(columns=['Avg wait from ED'])
            y = df_clean['Avg wait from ED']

            # Handle any remaining missing values in features
            X = X.fillna(X.mean())

            # Initialize and train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Get feature importances
            importances = pd.Series(model.feature_importances_, index=X.columns)
            top_5_features = importances.nlargest(5)

            # Return the results as is. 
            return top_5_features

        """

        result, code = self.agent.coder(system_prompt)      

        utils.assistant_message("code", code)
        utils.assistant_message("result", [result])

        return result      
            
            
    def main(self):
        with st.expander('Base Agent.', expanded=False):
            result = self.base()

        st.write(result)

        return result
