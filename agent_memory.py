import utils 
from agents import Agent, ReAct
import datetime
import json
import streamlit as st


chat_history_file = "message_history/daily_chat_log.json"

with open(chat_history_file, "r") as f:
    chat_history = json.load(f)

class MemoryAgent:
    def __init__(self, user_input):
        self.user_input = user_input 
        self.month, self.week, self.date = utils.get_current_time_components()

    def recall(self):
        system_prompt = """
        You are a memory agent. Your job is to look over the prompts and text chains between user and LLM from previous interactions. 

        You will be given a JSON which will have all the meessages between user and LLM in the current session and in previous days, weeks and months. 

        Time Information: 
        You will be given the month as an integer (1-12). 
        The week as an integer (1-52).
        And the date_day_key, which is defined as date_day_key = now.strftime("%m_%d_%Y_%A")

        If you write a date, write it in a clean manner: 
        So if you see 05_27_2025_Tuesday, rather write it as **Tuesday, May 27th**. 
        """

        user_prompt = self.user_input 

        user_prompt += f""""\n\n
        Here is the current date information for today.
        Month (1-12): {str(self.month)}
        Week (1-52): {str(self.week)}
        Date_day: {str(self.date)}
        \n\n
        """
        
        user_prompt += f"""
        Here is the conversation history: {str(chat_history)}
        """

        response = Agent(user_prompt).chat(system_prompt)
        utils.assistant_message('chat', response)

        return response

    def main(self):
        return self.recall()

