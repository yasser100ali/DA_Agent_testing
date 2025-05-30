import utils
from agents import Agent
import streamlit as st
import pandas as pd
import time
from agent_filter_data import filter_data
import json

# testing pdf 
#from pdf_generator import generate_pdf_reportlab

class DeepInsightsAgent:
    def __init__(self, user_input, local_var):
        self.user_input = user_input
        self.local_var = local_var
        self.dataframes_dict = local_var['dataframes_dict']
        self.model = "hf.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF:Q8_0"
        #self.model = "hf.co/bartowski/Qwen_QwQ-32B-GGUF:Q8_0"

    def deepinsights(self):
        orchestrator_job = """
            Take the user prompt and break it down to a series of tasks and assign each task to the following tool:
                a. data_processing_agent

            **WRITE 'data_processing_agent' EVERY TIME**
            When creating the plan, know that the subagent "data_processing_agent" has numpy, pandas and sklearn at their disposal so plan accordingly. 

            Follow this format: 

            Input: 
            "How could I reduce avg wait from ED?"
    
            Output: 
            I will outline a step by step plan to find out how I could best reduce the feature avg wait from ED.
            ```json
            {
                "1": {
                    "tool": "data_processing_agent",
                    "task": "Find the features that are most correlated with the feature avg wait from ED. Use pd.corr() for this. Get the distribution of Avg wait from ED (the min value, mean max, etc)."
                },
                "2": {
                    "tool": "data_processing_agent",
                    "task": "from the sklearn library use randomforest to get the most impactful features. Return the scores"
                },
                "3": {
                    "tool": "data_processing_agent",
                    "task": "Using the last output of most impactful features, verify it is statistically significant using statsmodels"
                }
                
            }
            ```

            Generally keep the tasks short and do around 3 or 4. But if it is a more broad question, then do more steps. 

            * If possible have one task where you use statsmodels to generate and test a hypothesis relevant to user prompt. 
            Also use sklearn if you are trying to find most impactful features (which you often will need to understand true relationships between the columns in the data.)
        """
        def get_prompt(agent, filtered_features):
            standard =  f"""

            You are a coder agent. Keep the code mostly short and concise, use as few lines as possible to accomplish each task. 
            
            1. Write complete self-contained code that includes all necessary imports
            2. You are given 'dataframes_dict' which contains all the files, pages (pages only applies to non-csv excel files), and feature names. To get access to a particular page, you would type 'dataframes_dict['excelsheetname.xlsx']['page'] or if it is a csv dataframes_dict['file.csv']
            3. Define a `main` function that encapsulates the task. The `main` function should:
            - Perform the requested task.
            - Return the result as a JSON
            4. Ensure the code is syntactically correct and ready to execute. Use markdown in your formatting: ```python import ... def main(): ... ```
            5. Use pandas, numpy or sklearn. Do not use libraries outside of these. 

            Correct spelling according to this list: {filtered_features}

            This list above, defines the **ONLY** valid files, sheets (for Excel), and columns/features you are permitted to access and use for this task. You **MUST** treat this as the definitive schema.
            DO NOT TRY TO ACCESS ANY FILES, PAGES, OR FEATURES OUTSIDE THIS.
            NEVER ASSUME ANYTHING THAT DOES NOT EXISTS IN THE LIST ABOVE. 

            You will be given the general user input and the specific task. The goal is the user_input but write your code mainly to the task with your orientation to the user_input. 

            Your outputs will be given to a seperate agent, so try to keep the outputs on the shorter side with dense information. 
            """
            
            system_prompts = {
                'data_processing_agent': standard + """
                    You are data_processing_agent.
                    
                    Your job is to use pandas for data analysis and manipulation, and sklearn for machine learning problems, according to the user request.
                    Use pandas to get relevant information, perform calculations, or prepare data.
                    If you have a pandas dataframe, make sure to return it as a JSON.
                    DO NOT return multiple items seperately, return as a list. So if you return to objects (a, b) return as [a, b]. 

                    Example 1: Data Analysis returning outputs as a JSON
                    input_prompt:
                    "From the pmpm average values page, give a table of Month and total PMPMs. Then in the same table take the Member months from 2025 forecast and multiply it by Total PMPMs from pmpm_average_values page to get me a new column 'Total Revenue' that is the product of these two."
                
                    output:
                    ```python
                    import pandas as pd
                
                    def main():
                        # Load the dataframes
                        pmpm_df = dataframes_dict['pmpm_and_2025_forecast.xlsx']['pmpm_average_values']
                        forecast_df = dataframes_dict['pmpm_and_2025_forecast.xlsx']['2025_forecast']
                
                        # Select required columns from each dataframe
                        # Assuming 'Month' in pmpm_df corresponds row-wise to the forecast_df
                        pmpm_table = pmpm_df[['Month', 'Total PMPMs']].head(len(forecast_df)) # Ensure lengths match if needed
                
                        # Get the member months
                        forecast_member_months = forecast_df[['Member Months']]
                
                        # Reset index if needed for clean concatenation/joining, or join on a common key if available
                        pmpm_table = pmpm_table.reset_index(drop=True)
                        forecast_member_months = forecast_member_months.reset_index(drop=True)
                
                        # Combine them into a new dataframe
                        combined_df = pd.concat([pmpm_table, forecast_member_months], axis=1)
                
                        # Calculate total revenue as the product of Total PMPMs and Member Months
                        # Ensure columns are numeric
                        combined_df['Total PMPMs'] = pd.to_numeric(combined_df['Total PMPMs'], errors='coerce')
                        combined_df['Member Months'] = pd.to_numeric(combined_df['Member Months'], errors='coerce')
                        combined_df['Total Revenue'] = combined_df['Total PMPMs'] * combined_df['Member Months']
                
                        # important to return as a JSON
                        return combined_df.to_json()
                    ```
                

                    Example 2: 
                    input_prompt: 
                    "Identify the most correlated features with average wait time using pd.corr() and return the top 5 features."

                    output: 
                    ```python
                    import pandas as pd
                    import plotly.express as px
                    
                    def main():
                        # Load the relevant dataframe
                        df = dataframes_dict['tabular_data.xlsx']['tabular_data']
                    
                        # Ensure 'Avg wait from ED' is numeric
                        df['Avg wait from ED'] = pd.to_numeric(df['Avg wait from ED'], errors='coerce')
                    
                        # Calculate statistics for Avg wait from ED
                        stats = {
                            'Min': df['Avg wait from ED'].min(),
                            'Max': df['Avg wait from ED'].max(),
                            'Mean': df['Avg wait from ED'].mean(),
                            'Median': df['Avg wait from ED'].median()
                        }
                        stats_df = pd.DataFrame([stats])
                        
                        # important to return as a json since this will be received by a seperate LLM. 
                        return stats_df.to_json()
                    ```
        
                    Once again you have 3 libraries to work with: 1. pandas 2. sklearn. Use according to your discretion. 

                """
            }

            return system_prompts[agent]

        agent = Agent(self.user_input, self.local_var)
        with st.expander('Thinking of plan...'):
            json_plan = agent.json_agent(orchestrator_job, self.model)


        plan = []
        st.write(utils.typewriter_effect('**Here is the plan that I have crafted.**'))
        for i, step in enumerate(json_plan.values()):
            task = step["task"]
            st.write(utils.typewriter_effect(f"Step {i + 1}"))
            st.write(utils.typewriter_effect(f"**{task}**"))
            plan.append(task)
        
        results = []

        problem_work = {
            "plan": plan,
            "plan execution": None,
            "report": None, 
            "report_visual": None,
            "conclusion": None
        }

        plan_execution = []
        
        features_list = str(utils.convert_to_features_list(self.dataframes_dict))
        features_list_json = utils.extract_json(features_list)
        word_count = len(features_list.split())
        st.write(word_count)

        if word_count > 150:
            input_features_list = filter_data(real_input, features_list)
            input_features_list_json = utils.extract_json(input_features_list)
        
        else:
            input_features_list = features_list
            input_features_list_json = features_list_json
        
        st.session_state['features_list_json'] = input_features_list_json
        
        problem_code = []
        previous_output = []

        with st.expander('Task Execution and Evidence Gathering', expanded=False):    
            for step_number, step in enumerate(json.values(), start=1):
                
                tool = step["tool"]
                task = step["task"]
                if tool not in ['data_processing_agent']: 
                    tool = 'data_processing_agent'
                    
                # new
                real_input = f"""
                Here is the whole plan: {str(json_plan)}  
                You are step {step_number}
                Specific Task (focus on this task only): {task}
                """

                # new
                if len(previous_output) > 0:
                    real_input += f"Here are the previous outputs:"
                    for number, output in enumerate(previous_output, start=1):
                        real_input += f"Output_{number}"
                        real_input += str(output)
                        real_input += "\n\n"

                system_prompt = get_prompt(tool, input_features_list)
        
                result, code = agent.coder(system_prompt, self.model, real_input)
                previous_output.append(result)

                # new
                if isinstance(result, pd.DataFrame):
                    result = result.to_json()

                if isinstance(result, dict):
                    st.write('true0')
                    df_to_display = pd.DataFrame.from_dict(result)
                    utils.show_output(pd.read_json(df_to_display))
                

                execution = (code, result)
                plan_execution.append(execution)
                problem_code.append(code)

               
                results.append(result)
            
        report_results = utils.filter_figures(results)

        reporter_input = f"User Input: {self.user_input}   Results and Evidence: {str(report_results)}"


        reporter_role = """
        Look over these results answer the user prompt based on these findings. Keep your response efficient and include a thourough amount of evidence to make your report. Make it professional. 
        If possible make a **single definitive claim** that answers the user's questions and condenses the information you found into a single sentence. Outline "Claim" before you say this. 
        
        If you have statistics and numbers, make sure to use those in your analysis.

        Do NOT write a conclusion sentence, that is the job of a different agent. Your job is merely the introduction and the main body of the report. 
        
        Make sure to use markdown to specify it is the Report and emphasize key points. 

        Make it professional, answer the user prompt by making your claim and the report (body paragraphs and bullet points) should use stats to support that claim or claims. 
        Make it efficient, but give solid and a decent amount of evidence to really support your claim and lay out a good argument for why you are making your particular claim. 

        If a table helps concisely conveying information to the user, then have a table. Have a SINGULAR table if you choose to have one, NOT more than 1. 

        Also write out the features names in a clean way (so if you see 'AVG_WAIT_FROM_ED' write out 'Avg Wait from ED'. Do not retain the underscores (_) and capitalizations.) 
        But for these same key features, bolden them using markdown. 
        """

        report = agent.reporter(reporter_role, self.model, reporter_input)
        
        def report_coder(user_input, report, features_list, code):
            system_prompt = """
            Look over the report and user input and come up with a python script where you create a visual (could do just one, or a 2x1 or a 2x2) to accomdate the report. 
            This must be clean with a title and potentially a note. It should be related to the report generated by a previous agent.
            
            follow the format of the codes you are given and generate a chart (could be one, 2 or 4) that caputres that most important aspects of the report and code. 
            
            In general, stick to 4 charts, unless the user prompt is very simple and can be explained away in 1 or 2. 
            """
            user_input_prompt = f"""
            Create visuals to go along with my report. Capture the essence of the report based on the user input and report. 

            Here is the user_input that triggered the report: {user_input}
            
            Here is the report itself: {report}
            
            When writing your code you will get your data from a dictionary already give to you 'dataframes_dict', spell exactly as shown: {features_list}
            
            Here is all the code that went into finding the outcomes found in the report: {code}
            
            When writing the code follow the same format as above, ```python 
            import matplotlib.pyplot as plt 
            import pandas as pd
            
            def main():
                df = dataframes_dict['...']
                ...
                return fig
    

            DO NOT use seaborn, use matplotlib (this is very important.)
            """
            
            result, code = agent.coder(system_prompt, "", user_input_prompt, hide_code=True)
            utils.show_output(result)
            
            return result

        with st.spinner('Generating Chart'):

            report_visual = report_coder(self.user_input, report, features_list, problem_code)  
  

        conclusion_system_prompt = """
        Look over the main point of the user and the general outputs of the report. Generate a short (one or two sentences) conclusion that takes the main points of the report and nicely concludes it.
        Make it concise and professional, reiterating the main things found by the reporter agent. 
        
        Make sure to use markdown and title the conclusion. 
        """
        conclusion_input_prompt = f"""
        Here is the report you are generating the conclusion for: {report}. 
        """
        conclusion = agent.reporter(conclusion_system_prompt, "", conclusion_input_prompt)
    
        problem_work["plan execution"] = plan_execution
        problem_work["report"] = report
        problem_work["report_visual"]  = report_visual
        problem_work["conclusion"] = conclusion
        
        utils.assistant_message('deepinsights', problem_work)
        
        return report
        

    def main(self):
        report = self.deepinsights()
        return report 
    