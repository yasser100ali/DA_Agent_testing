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
        

    def plan_and_reason(self):
        
        system_prompt = """
        You are an autonomous data analyst agentic system. Your have a variety of agents at your disposal,

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

            
        Your job is to take the user prompt and to assign the user prompt to a singular agent or to devise a step by step plan to satisy everything the user asked for. 

        Example of a singular agent call would look like this: 

        user_input: "pull tables related to financial numbers"

        your output:
        ```json 
        {
            "agent": "pull_table"   
        }
        ```

        user_input: "write a report on the financial numbers from the last quarter"
        your output: 
        ```json
        {
            "agent": "deepinsights"
        }
        ```
        

        Examples of multiple agent call will look like the following: 

        user_input: "Pull tables relating to finacial numbers from the database, then write a report on the numbers from the last quarter, finally look up and see if I have anything on my calendar for the rest of the week.
        your output: 

        ```json
        {
            1: {
                "agent": "pull_table",
                "instructions": "Pull tables related to financial numbers from the database"   
            },
            2: {
                "agent": "deepinsights",
                "instructions": "Write a report on the financial numbers from the last quarter"
            },
            3: {
                "agent": "secretary",
                "instructions": "Check the calendar for the remainder of the week and return details to user"
            }
        }
        ```

        Notice that in the first few examples, a single agent would suffice, whereas in the last example, the job needed the help of multiple agents. Bef
        """

        agent_plan = Agent(self.user_input).json_agent(system_prompt)

        return agent_plan
    
    def act(self):
        agent_plan = self.plan_and_reason()

        if len(agent_plan) == 1:
            agent_name = agent_plan["agent"]

            if 