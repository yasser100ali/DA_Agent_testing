import utils.utils as utils
from secretary_ai_functions import read_emails, send_email, list_upcoming_events, create_calendar_event
from agents.agents import Agent
import json 
import streamlit as st
import time

class SecretaryAI:
    def __init__(self, user_input):
        self.user_input = user_input
        self.local_var = {
            'read_emails': read_emails,
            'send_email': send_email,
            'list_upcoming_events': list_upcoming_events,
            'create_calendar_event': create_calendar_event
        }

    def orchestrator(self):
        system_prompt = """
        Your job is to assign the task to either connector_agent or mail_and_calendar_agent. 
        If the task involves connecting the user to a seperate person, then call connector_agent. 
        If the task involves reading or writing emails, or reading or scheduling something on calendar, then call mail_and_calendar_agent. 

        Example1: 
        "I am trying to build a multi-agent data analyst system, who could I talk to do this?" 
        
        Response: 
        ```json
        {
            "agent": "connector_agent"
        }
        ```

        Example2: 
        "Check my email and see if I got any messages from John"
        Response: 
        ```json
        {
            "agent": "mail_and_calendar_agent"
        }
        ```
        """

        agent = Agent(self.user_input, {})
        json = agent.json_agent(system_prompt, "")
        return json


    def mail_and_calendar_agent(self): # user coder agent to write function which will decide which api function to use. -> might change this to an MCP style 
        system_prompt = """
        You are a function caller agent. Your job is to take the query and decide which of the 4 functions to use:
        
        Functions and parameters. 
        {
            { 
                "function": "read_emails",
                "paramters": { 
                    "days": 7,  
                    "max_results": 100, 
                    "query"="in:inbox"
                    #These are the default values. 
                }
            },
            {
                "function": "send_email",
                "paramters": { 
                    "to": "example@gmail.com",  
                    "subject": "(subject line)", 
                    "body"="This would be an example of what would go in the body "
                }
                # no default values here, must write out each of the 3 parameters
                
            },
            {
                "function": "list_upcoming_events",
                "paramters": { 
                    "days": 7, # default value  
                    "max_results": 10, # default value 
                }
                # no default values here, must write out each of the 3 parameters
                
            },
            {
                "function": "create_calendar_event",
                "paramters": { 
                    "summary": "summary (str): The title of the event.",  
                    "start_time_str": "2025-05-05T10:00:00-07:00", # in RFC3339 format  
                    "end_time_str"="2025-05-05T11:00:00-07:00" # in RFC3339 format
                }
                # no default values here, must write out each of the 3 parameters
            },

        }
        

        
        Format **MUST FOLLOW ACCORDINGLY**
        You are to output your answer in the following way:
        user_input: "send an email to (example@gmail.com) saying Sounds good, let's review the progress next monday. then ask if 2:30 is good."
        {
            "function": "send_email",
            "parameters": {
                "to": "example@gmail.com",
                "subject": "Reviewing project progress",
                "body": "Sounds good, let's review the progress next Monday. Does 2:30 sound good?"
            }
        }

        """
        json_output = Agent(self.user_input, self.local_var).json_agent(system_prompt, "dummy_model", is_visible=True)
        utils.assistant_message('json_output', json_output)
        function_output = self.local_var[json_output["function"]](**json_output["parameters"])
        print("\n\nOutput\n\n")
        print(function_output)
        chat_prompt = f"""
        Your task is to relay the following information to the user based on what they asked for: {str(function_output)}
        **IMPORTANT**. If creating an email simply show the email you created and ask if there is anything else. Same goes for creating event on calendar.
        If writing an email or creating event on calendar simply show what you sent or created and ask if there is anything else for you to do. 
        Example when sending an email. 
        "Ok I've sent the email to (infer first name if possible). Here is what I wrote: (show message). Is there anything else I could do for now?" 
        BE CONCISE. 
        """
        content = Agent(self.user_input, self.local_var).chat(chat_prompt, "dummy_model")
        utils.assistant_message('chat', content)

        return content 

    def connector_agent(self):
        system_prompt = """
        Look over this file structure, it represents the Kaiser organization. 

        Based on the user input, help the user get connected with the right person and the right team. It may be multiple people (usually will be) and it maybe be multiple teams (rarer but will happen if you are unsure or think a collaberative effort is necessary). 

        Your job is to essentially bridge the gap between different members of the organization and relay what other people are doing so that the user could collaberate with other teams, get help if they are confused and a different team or member may be of help, or if they are just curious what other people do. 

        Return the person(s) and team(s) that could help along with their contact and a short reason why you chose them. 

        When you write out the names bolden them and Write without the underscores. So if name is "JOHN_SMITH", write as **John Smith**. 
        
        Make it short and to the point, give a brief reason why. 
        """

        filename = 'connections_ai_data/employee_data.json'
        with open(filename, 'r') as f:
            organization = json.load(f)

        user_input = self.user_input 
        user_input += f"\nHere is the organization data: \n{str(organization)}"


        agent = Agent(user_input, {})
        agent_answer = agent.chat(system_prompt, "")


        utils.assistant_message('chat', agent_answer)
        return agent_answer

    def main(self):
        subagent = self.orchestrator()

        if subagent['agent'] == 'mail_and_calendar_agent':
            return self.mail_and_calendar_agent()
        
        elif subagent['agent'] == 'connector_agent': 
            return self.connector_agent()
    
        else:
            return self.main()