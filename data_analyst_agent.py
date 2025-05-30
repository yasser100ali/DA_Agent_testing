from agent_base import BaseAgent
from agent_deepinsights import DeepInsightsAgent
from agent_chat import ChatAgent
from agent_secretary import SecretaryAI
from agent_sql_react_testing import sqlAgentReAct
from agent_memory import MemoryAgent
from agent_sql import sqlAgent 
from agents import Agent
import utils
import streamlit as st

class DataAnalystAgent:
    def __init__(self, user_input, local_var):
        self.user_input = user_input
        self.local_var = local_var
        self.agents = {
            'base': BaseAgent,
            'deepinsights': DeepInsightsAgent,
            'chat': ChatAgent,
            'secretary': SecretaryAI,
            'sql': sqlAgentReAct,
            'memory': MemoryAgent,
            'pull_table': sqlAgent
        }
        
        self.model = "hf.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF:Q8_0"

    def operator(self):
        if len(st.session_state.dataframes_dict) > 0:
            system_prompt = """
                Your task is to read the user prompt and assign it to the appropriate agent. 

                You have 7 agents: "base", "deepinsights", "chat", "secretary", "memory", "sql", and "pull 

                1. "base"
                    - for straightforward and simple data analysis tasks, typically singular in approach
                    - example prompts could be 
                        - 'Give a scatter plot of (features here)'
                        - 'Return the following table ...'
                        - often times will ask for a simple thing (visualization table etc) 
                        - if a prompt could be answered in a single script, then this is the go to. 
                2. "deepinsights"
                    - this is when the user asks for a report. If a prompt requires multiple steps and a report at the end, then go here. 
                    - most of the time the user will ask for a report or something along these lines. 
                3. "chat"
                    - when none of the agents are required and the prompt from the user requires more of a general chat response. 
                    - often this will ask general frequently asked questions like 'what can you do', or 'how do you work', or 'what are some general tips for...' or 'hi who are you', etc. 
                4. "secretary"
                    - for when a question is asked regarding sending emails, reading email, scheduling event on calendar or reading calendar
                    - for when the user asks who they could get help from within the organization
                    - for when the user may need to be connected to someone within the organization
                    - might ask "What can you tell me about team "X" (team name here) or person "Y", what do they do? 
                    - might say "I am having a problem with (problem here), who do you know that could help me with this task?"  
                5. "sql" 
                    - when there is a request to get data from a database using sql
                    - When asked to go through a database
                    - When asked to simply look through a database
                6. "memory"
                    - when the user is asking you to recall a previous text-chain or a previous conversation. 
                    - Whenever asked to look into the past about something that may be from a previous conversation
                7. "pull_table"
                    - when asked to actually "pull" the table from a database. 
                    - Here the user wants the table from the database to be imported into their files. This is similar to 5 "sql" except here the user is asking for the table to be imported or pulled. 

                give your answer in the following format 
                example when "base" is the chosen agent. 
                ```json
                {
                    "agent": "base"
                }
                ```

               
            """
            
        else: 
            system_prompt = """
                Your task is to read the user prompt and assign it to the appropriate agent. 

                You have 5 agents: "chat", "secretary", and "sql", "memory", and "pull_table"

                1. "chat"
                    - when none of the agents are required and the prompt from the user requires more of a general chat response. 
                    - often this will ask general frequently asked questions like 'what can you do', or 'how do you work', or 'what are some general tips for...' or 'hi who are you', etc. 
                2. "secretary"
                    - for when a question is asked regarding sending emails, reading email, scheduling event on calendar or reading calendar
                    - for when the user asks who they could get help from within the organization
                    - for when the user may need to be connected to someone within the organization
                    - might ask "What can you tell me about team "X" (team name here) or person "Y", what do they do? 
                    - might say "I am having a problem with (problem here), who do you know that could help me with this task?"  
                3. "sql" 
                    - when there is a request to pull data from a database. 
                    - When asked to go through a database
                4. "memory"
                    - when the user is asking you to recall a previous text-chain or a previous conversation. 
                    - Whenever asked to look into the past about something that may be from a previous conversation
                5. "pull_table"
                    - when asked to actually "pull" the table from a database. 
                    - Here the user wants the table from the database to be imported into their files. This is similar to 5 "sql" except here the user is asking for the table to be imported or pulled. 
                    

                give your answer in the following format 
                ```json
                {
                    "agent": "secretary"
                }
                ```
            """
        subagent = Agent(self.user_input, self.local_var).json_agent(system_prompt, self.model)['agent']

        return subagent  
    
    def main(self):
        
        subagent_name = self.operator()
        
        # if subagent_name != 'chat':
        #     st.write(utils.typewriter_effect(f'This task has been assigned to: **{subagent_name}** agent.'))

        if subagent_name in ["chat", "deepinsights"]: 
            params = {
                "user_input": self.user_input,
                "local_var": self.local_var
            }
        
        else:
            params = {
                "user_input": self.user_input
            }

        subagent = self.agents[subagent_name](**params)
        subagent.main()