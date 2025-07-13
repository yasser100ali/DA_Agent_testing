import utils.utils as utils
from agents.agents import AsyncAgent, Agent
import streamlit as st
import pandas as pd
import asyncio
import time 

class DeepInsights:
    def __init__(self, user_input, local_var={}):
        self.user_input = user_input
        self.local_var = local_var
        self.async_agent = AsyncAgent(user_input, local_var=local_var)
        self.agent = Agent(user_input, local_var=local_var)
        self.features_list = str(utils.convert_to_features_list(st.session_state['dataframes_dict']))

    def generate_plan(self):

        # generates plan which will be fed into the execute_plan function
        system_prompt = """
            Take the user prompt and break it down into a series of tasks and assign each task to the following tool:
                a. data_processing_agent

            **WRITE 'data_processing_agent' EVERY TIME**
            When creating the plan, know that the subagent "data_processing_agent" has numpy, pandas and sklearn at their disposal so plan accordingly.

            --- Brevity and Output Operator ---
            **Your primary goal is to create a plan that results in CONCISE outputs.** Each task you create should instruct the `data_processing_agent` to return only the most essential, summarized, or aggregated information. Do not create tasks that ask the agent to return raw or unfiltered tables.

            - GOOD TASK: "Find the top 5 most correlated features with 'Avg Wait Time' and return only those 5 features and their correlation scores."
            - GOOD TASK: "Calculate the total revenue and average monthly revenue and return only these two numbers."
            - BAD TASK: "Show me the table of PMPM values and Member months."
            - BAD TASK: "Get all the player statistics."
            ------------------------------------

            Follow this format: 

            Input: 
            "How could I reduce avg wait from ED? Write a report on this"
        
            Output: 
            I will outline a step by step plan to find out how I could best reduce the feature avg wait from ED.
            ```json
            {
                "1": {
                    "tool": "data_processing_agent",
                    "task": "Find the top 10 features that are most correlated with the feature 'avg wait from ED'. Use pd.corr() for this and return only the feature names and their correlation scores."
                },
                "2": {
                    "tool": "data_processing_agent",
                    "task": "From the sklearn library, use a RandomForestRegressor to get the most impactful features on 'avg wait from ED'. Return a list of the top 5 features and their importance scores."
                },
                "3": {
                    "tool": "data_processing_agent",
                    "task": "Using the list of most impactful features, generate a hypothesis on whether their combined effect is statistically significant. Use a regression model from statsmodels to test this and return the p-values and R-squared value."
                }   
            }
            ```

            Generally keep the tasks short and do around 4 to 7. But if it is a more broad question, then do more steps. 

            * If possible have one task where you use statsmodels to generate and test a hypothesis relevant to user prompt. 
            Also use sklearn if you are trying to find most impactful features (which you often will need to understand true relationships between the columns in the data.)

            If the user asks to "generate a report" of some kind, simply gather evidence that you think may help write the report for the user. Do not have a step to actually write a report; that is handled by a separate agent. Your job is simply to direct a plan that would gather the evidence, not tell the next agent to write a report. 
        """
        
        json_plan = self.agent.json_agent(system_prompt, user_input=self.user_input)
        st.status("Plan Generated, moving to Plan Execution Stage")

        return json_plan

    # helper function
    def _get_prompt(self, task, features_list):
        standard =  f"""

        You are a coder agent. Keep the code mostly short and concise, use as few lines as possible to accomplish each task. 
        
        1. Write complete self-contained code that includes all necessary imports
        2. You are given 'dataframes_dict' which contains all the files, pages (pages only applies to non-csv excel files), and feature names. To get access to a particular page, you would type 'dataframes_dict['excelsheetname.xlsx']['page'] or if it is a csv dataframes_dict['file.csv']
        3. Define a `main` function that encapsulates the task. The `main` function should:
        - Perform the requested task.
        - Return the result as a JSON
        4. Ensure the code is syntactically correct and ready to execute. Use markdown in your formatting: ```python import ... def main(): ... ```
        5. Use pandas, numpy or sklearn. Do not use libraries outside of these. 

        Correct spelling according to this list: {features_list}

        This list above, defines the **ONLY** valid files, sheets (for Excel), and columns/features you are permitted to access and use for this task. You **MUST** treat this as the definitive schema.
        DO NOT TRY TO ACCESS ANY FILES, PAGES, OR FEATURES OUTSIDE THIS.
        NEVER ASSUME ANYTHING THAT DOES NOT EXISTS IN THE LIST ABOVE. 

        You will be given the general user input and the specific task. The goal is the user_input but write your code mainly to the task with your orientation to the user_input. 

        Focus on getting a thorough amount of evidence to answer the original user_prompt: {self.user_input}
        Don't get too much frivelous data, get a decent amount of valuable data that will later be used to build a report. Your whole job is to gather evidence that will later produce a report. 
        This will later be fed to a seperate agent that will build the report so your outputs DENSE. 
        """
        
        system_prompts = {
            'data_processing_agent': standard + """
                You are data_processing_agent.
                
                Your job is to use pandas for data analysis and manipulation, and sklearn for machine learning problems, according to the user request.
                Use pandas to get relevant information, perform calculations, or prepare data.
                If you have a pandas dataframe, make sure to return it as a JSON.
                DO NOT return multiple items seperately, return as a list. So if you return to objects (a, b) return as [a, b]. 
                Example 1: Data Analysis returning a concise JSON summary
                        input_prompt:
                        "From the pmpm average values page, give a table of Month and total PMPMs. Then in the same table take the Member months from 2025 forecast and multiply it by Total PMPMs from pmpm_average_values page to get me a new column 'Total Revenue' that is the product of these two."
                        
                        output:
                        ```python
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
                            
                            # Return the small summary dictionary as a JSON string. This is very token-efficient.
                            return json.dumps(summary, indent=4)
                        ```
                        

                        Example 2: Returning a small, targeted slice of data
                        input_prompt: 
                        "Identify the most correlated features with average wait time using pd.corr() and return the top 5 features."

                        output: 
                        ```python
                        import pandas as pd
                        
                        def main():
                            # Load the relevant dataframe
                            df = dataframes_dict['tabular_data.xlsx']['tabular_data']
                            
                            # Create a list of numeric columns for correlation calculation
                            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                            
                            # Calculate the correlation matrix for numeric columns only
                            corr_matrix = df[numeric_cols].corr()
                            
                            # Get the correlations with the target variable 'Avg wait from ED'
                            # Drop the self-correlation (it will be 1.0)
                            wait_time_corr = corr_matrix['Avg wait from ED'].drop('Avg wait from ED')
                            
                            # Get the top 5 most correlated features by absolute value
                            top_5_features = wait_time_corr.abs().nlargest(5)
                            
                            # The result is a small pandas Series. Returning this as JSON is concise and dense.
                            return top_5_features.to_json()
                        ```

                        Once again you have 3 libraries to work with: 1. pandas 2. sklearn. Use according to your discretion. 

            """
        }

        return system_prompts[task]

    async def execute_plan(self, plan):
        # executes the plan developed by generate_plan function. Each LLM will run in parallel for significant time speed improvements. 
        start_time = time.time()

        if st.session_state.dataframes_dict:
            
            features_list_json = utils.extract_json(self.features_list)
            st.session_state['features_list_json'] = features_list_json

        for step in plan: 
            plan[step]["system_prompt"] = self._get_prompt(plan[step]["tool"], self.features_list)

        tasks = [
            self.async_agent.coder_agent(
                system_prompt=step["system_prompt"], user_input=step["task"]
            ) for step in plan.values()
        ] 

        # each result in results is a tuple -> (result, code)
        results = await asyncio.gather(*tasks) 


        end_time = time.time()
        total_time = end_time - start_time

        print(f"\nTotal time to finish {len(plan)} prompts: {total_time:.2f} seconds\n")

        return results
    
    def write_report(self, execute_plan_findings):
        # takes outputs of the stage above and generates a report that synthesizes findings in a coherent manner
        reporter_input = f"User Input: {self.user_input}   Results and Evidence: {str(execute_plan_findings)}"

        reporter_system_prompt = """
        Look over these results answer the user prompt based on these findings. Keep your response efficient and include a thorough amount of evidence to make your report. Make it professional. 
        If possible make a **single definitive claim** that answers the user's questions and condenses the information you found into a single sentence. Outline "Claim" before you say this. 
        
        If you have statistics and numbers, make sure to use those in your analysis.

        Do NOT write a conclusion sentence, that is the job of a different agent. Your job is merely the introduction and the main body of the report. 
        
        Make sure to use markdown. For the title, have it be something that relates to the report. 

        Make it professional, answer the user prompt by making your claim and the report (body paragraphs and bullet points) should use stats to support that claim or claims. 
        Make it efficient, but give solid and a decent amount of evidence to really support your claim and lay out a good argument for why you are making your particular claim. 

        If a table helps concisely conveying information to the user, then have a table. Have a SINGULAR table if you choose to have one, NOT more than 1. 

        Also write out the features names in a clean way (so if you see 'AVG_WAIT_FROM_ED' write out 'Avg Wait from ED'. Do not retain the underscores (_) and capitalizations.) 
        But for these same key features, bolden them using markdown. 

        Make it efficient, but explain things in paragraphs so that the user easily understands. Make it a professional report. 
        """
    
        report = self.agent.reporter(reporter_system_prompt, reporter_input)

        return report 
    
    def generate_charts_for_report(self, execute_plan_findings, report):
        with st.spinner("Generating Chart"):
            # generates charts for report to give a visual representation of what report is attempting to express
            system_prompt = """
            Look over the report and user input and come up with a python script where you create a visual (could do just one, or a 2x1 or a 2x2) to accomdate the report.
            This must be clean with a title and potentially a note. It should be related to the report generated by a previous agent.

            Follow the format of the codes you are given and generate a chart (or multiple subplots) that captures the most important aspects of the report and code.

            In general, stick to 4 charts, unless the user prompt is very simple and can be explained away in 1 or 2.

            Use plotly. 
            """
            user_input_prompt = f"""
            Create interactive visuals to go along with my report. Capture the essence of the report based on the user input and report.

            Here is the user_input that triggered the report: {self.user_input}

            Here is the report itself: {report}

            When writing your code you will get your data from a dictionary already give to you 'dataframes_dict', spell exactly as shown: {self.features_list}

            Here is the data used to get the report: {str(execute_plan_findings)}

            When writing the code follow the same format as above, ```python
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd

            def main():
                df = dataframes_dict['...']
                # Example: fig = px.bar(df, x='Category', y='Values', title='Example Chart')
                ...
                return fig


            Please use the plotly library (plotly.express or plotly.graph_objects) for all visualizations as it is more interactive.
            """

            chart_for_report, _ = self.agent.coder(system_prompt, user_input_prompt, hide_code=True)
            
        utils.show_output(chart_for_report)

        return chart_for_report

    def write_conclusion(self, report):
        # adds a nice finishing touch and makes a small conclusion of the report
        conclusion_system_prompt = """
        Look over the main point of the user and the general outputs of the report. Generate a short (one or two sentences) conclusion that takes the main points of the report and nicely concludes it.
        Make it concise and professional, reiterating the main things found by the reporter agent. 
        
        Make sure to use markdown and title the conclusion. 
        """
        conclusion_input_prompt = f"""
        Here is the report you are generating the conclusion for: {report}. 
        """

        conclusion = self.agent.reporter(conclusion_system_prompt, conclusion_input_prompt)
        
        return conclusion

    async def main(self):
        plan = self.generate_plan()
        plan_findings = await self.execute_plan(plan)
        report = self.write_report(plan_findings)
        chart_for_report = self.generate_charts_for_report(plan_findings, report)
        conclusion = self.write_conclusion(report)

        deepinsights_dict = {
            "plan_findings": plan_findings,
            "report": report,
            "chart_for_report": chart_for_report,
            "conclusion": conclusion
        }   

        utils.assistant_message('deepinsights', deepinsights_dict)

        return deepinsights_dict
    


    