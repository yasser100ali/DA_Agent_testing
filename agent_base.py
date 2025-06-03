import utils
from agents import Agent
from agent_filter_data import filter_data
import streamlit as st
import time


class BaseAgent:
    def __init__(self, user_input, local_var):
        self.user_input = user_input
        self.local_var = local_var
        self.dataframes_dict = local_var['dataframes_dict']
        self.model = "not_needed" # fix this 
        
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
                    # Retrieve dataframes
                    membership_df = dataframes_dict['membership_predictions_0221.csv']
                    pmpm_df = dataframes_dict['pmpm_and_2025_forecast.xlsx']['pmpm_commercial_by_location']

                    # Filter membership data for Antioch and Manteca (Commercial line of business)
                    filtered_membership = membership_df[
                        (membership_df['MJR_AREA_NM'].isin(['ANTIOCH', 'MANTECA'])) & 
                        (membership_df['LINE_OF_BUSINESS'] == 'COMMERCIAL')
                    ]

                    # Get predicted mean values (assuming we want the mean prediction)
                    predicted_means = filtered_membership.groupby('MJR_AREA_NM')['PREDICTED_MEAN'].mean()

                    # Get PMPM values for Antioch and Manteca
                    pmpm_data = pmpm_df[['DATE', 'ANTIOCH', 'MANTECA']]

                    # Calculate revenue (PMPM * predicted_mean)
                    revenue_data = pmpm_data.copy()
                    revenue_data['ANTIOCH_REVENUE'] = revenue_data['ANTIOCH'] * predicted_means['ANTIOCH']
                    revenue_data['MANTECA_REVENUE'] = revenue_data['MANTECA'] * predicted_means['MANTECA']

                    # Melt the dataframe for plotting
                    plot_data = revenue_data.melt(
                        id_vars=['DATE'], 
                        value_vars=['ANTIOCH_REVENUE', 'MANTECA_REVENUE'],
                        var_name='LOCATION', 
                        value_name='REVENUE'
                    )

                    # Clean up location names
                    plot_data['LOCATION'] = plot_data['LOCATION'].str.replace('_REVENUE', '')

                    # Create bar chart
                    fig = px.bar(
                        plot_data,
                        x='DATE',
                        y='REVENUE',
                        color='LOCATION',
                        barmode='group',
                        title='Commercial Revenue for Antioch and Manteca Over the Year',
                        labels={'DATE': 'Month', 'REVENUE': 'Revenue ($)', 'LOCATION': 'Location'},
                        text='REVENUE'
                    )

                    # Format the plot
                    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        yaxis_tickprefix='$',
                        yaxis_tickformat=',.0f',
                        uniformtext_minsize=8,
                        uniformtext_mode='hide'
                    )

                    return fig

                ```
                """,

                'analysis': standard + """
                You are data_analyst_agent
                Your job is to use pandas to get statistics according to the user request.
                Return your results in a pandas dataframe. 

                Refrain from generating charts as that is not your responsibility, return the requested items as ONLY a pandas dataframe. 

                example:
                input_prompt:  
                From the pmpm average values page, give a table of Month (metric="Month", page="pmpm_average_values", file="pmpm_and_2025_forecast.xlsx") and total PMPMs
                (metric="Total PMPMs", page="pmpm_average_values", file="pmpm_and_2025_forecast.xlsx"). Then in the same table take the Member months from 2025 forecast (metric="Member 
                Months", page="2025_forecast", file="pmpm_and_2025_forecast.xlsx") and multiply it by Total PMPMs from pmpm_average_values page (metric="Total PMPMs", page="pmpm_average_values", file="pmpm_and_2025_forecast.xlsx") to get me a new column 'Total Revenue' that is the product of these two.
                
                output: 
                ```python
                import pandas as pd
                
                def main():
                    # Load the dataframes
                    pmpm_df = dataframes_dict['pmpm_and_2025_forecast.xlsx']['pmpm_average_values']
                    forecast_df = dataframes_dict['pmpm_and_2025_forecast.xlsx']['2025_forecast']
                
                    # Select required columns from each dataframe
                    pmpm_table = pmpm_df[['Month', 'Total PMPMs']]
                
                    # Get the first 12 rows for member months (assuming data is chronological)
                    forecast_member_months = forecast_df.head(12)[['Member Months']]
                
                    # Combine them into a new dataframe
                    combined_df = pd.concat([pmpm_table, forecast_member_months], axis=1)
                
                    # Calculate total revenue as the product of Total PMPMs and Member Months
                    combined_df['Total Revenue'] = combined_df['Total PMPMs'] * combined_df['Member Months']
                
                    return combined_df
                ```

                """,
                
                'machine_learning': standard + """

                """
            }
            
            sub_agent = json['agent']
            st.write(utils.typewriter_effect(f'Problem Type: **{sub_agent}**'))
            job = system_prompts[sub_agent]

            result, code = agent.coder(job, self.model)        
            utils.show_output(result)

            utils.assistant_message("code", code)
            utils.assistant_message("result", [result])

            return result        
        
        else:
            st.write('Trying again')
            self.base()
            
            
    def main(self):
        with st.expander('Base Agent. Code & Work.', expanded=True):
            result = self.base()
            

        return result