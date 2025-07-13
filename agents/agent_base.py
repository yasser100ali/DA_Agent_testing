import utils.utils as utils
from agents.agents import Agent
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
        features_list = str(utils.convert_to_features_list(self.dataframes_dict)) 
        features_list_json = utils.extract_json(features_list)
        st.session_state['features_list_json'] = features_list_json

        system_prompt =  f"""

        You are a coder agent. Keep the code mostly short and concise, use as few lines as possible to accomplish each task. 
        
        1. Write complete self-contained code that includes all necessary imports
        2. You are given 'dataframes_dict' which contains all the files, pages (pages only applies to non-csv excel files), and feature names. To get access to a particular page, you would type 'dataframes_dict['excelsheetname.xlsx']['page'] or if it is a csv dataframes_dict['file.csv']
        3. Define a `main` function that encapsulates the task. The `main` function should:
        - Perform the requested task.
        - Retrun result as a pandas dataframe or a plotly visual chart. 
        4. Ensure the code is syntactically correct and ready to execute. Use markdown in your formatting: ```python import ... def main(): ... ```
        5. Use pandas, numpy or sklearn. Do not use libraries outside of these. 

        Correct spelling according to this list: {features_list}

        This list above, defines the **ONLY** valid files, sheets (for Excel), and columns/features you are permitted to access and use for this task. You **MUST** treat this as the definitive schema.
        DO NOT TRY TO ACCESS ANY FILES, PAGES, OR FEATURES OUTSIDE THIS.
        NEVER ASSUME ANYTHING THAT DOES NOT EXISTS IN THE LIST ABOVE. 

        You will be given the general user input and the specific task. The goal is the user_input but write your code mainly to the task with your orientation to the user_input. 
        Don't get too much frivelous data, get a decent amount of valuable data that will later be used to build a report. Your whole job is to gather evidence that will later produce a report. 
        This will later be fed to a seperate agent that will build the report so your outputs DENSE. 
        """

        system_prompt += """
        You are data_processing_agent.

        Your job is to use pandas for data analysis and manipulation, plotly for creating interactive visualizations, and sklearn for machine learning problems. Your code output must always be in the format def main(): ... return result.

        You have 3 libraries to work with:

        pandas: Use for data manipulation, calculations, filtering, or when the user asks for a table. Return data as a pandas dataframe. 

        plotly: Use when the user asks for any kind of chart or visual (e.g., "plot", "chart", "graph", "visualize"). Final line here should be "return fig" 

        sklearn: Use for machine learning tasks like finding feature importance, making predictions, clustering, or classification.

        DO NOT return multiple items separately; if you have multiple outputs, return them as a list (e.g., return [a, b]).

        EXAMPLES. 

        example 1:

        input_prompt:
        "From the pmpm average values page, give a table of Month and total PMPMs. Then in the same table take the Member months from 2025 forecast and multiply it by Total PMPMs from pmpm_average_values page to get me a new column 'Total Revenue' that is the product of these two."
        import pandas as pd
        import json

        def main():
            # Load the dataframes
            pmpm_df = dataframes_dict['pmpm_and_2025_forecast.xlsx']['pmpm_average_values']
            forecast_df = dataframes_dict['pmpm_and_2025_forecast.xlsx']['2025_forecast']

            # Perform the calculation as requested
            combined_df = pmpm_df[['Month', 'Total PMPMs']].copy()
            combined_df['Member Months'] = forecast_df['Member Months']
            combined_df['Total PMPMs'] = pd.to_numeric(combined_df['Total PMPMs'], errors='coerce')
            combined_df['Member Months'] = pd.to_numeric(combined_df['Member Months'], errors='coerce')
            combined_df['Total Revenue'] = combined_df['Total PMPMs'] * combined_df['Member Months']

            # Instead of returning the whole table, create a concise summary.
            summary = {
                'total_revenue_all_months': combined_df['Total Revenue'].sum(),
                'average_monthly_revenue': combined_df['Total Revenue'].mean(),
                'highest_revenue_month': combined_df.loc[combined_df['Total Revenue'].idxmax()].to_dict(),
                'note': "Successfully calculated Total Revenue. Provided are key summary statistics."
            }

            # convert to pandas dataframe
            summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])

            # Return the new DataFrame.
            return summary_df


        example 2: 
        input_prompt:
        "Create an interactive bar chart showing the 'Bed TAT' for the top 5 'Primary Payers'."    
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
