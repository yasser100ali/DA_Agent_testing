import utils 
from agents import Agent, ReAct
import datetime
import json
import streamlit as st


chat_history_file = "message_history/daily_chat_log.json"

with open(chat_history_file, "r") as f:
    chat_history = json.load(f)

def trim_chat_history(current_chat_history, max_messages_per_day=15):
    """
    Alters the chat_history in place to keep only the first
    `max_messages_per_day` total messages for each day.
    Each "turn" (e.g., {"0": [...]}) can contain multiple messages.
    The function will truncate the list of messages within the last turn  
    if necessary to meet the max_messages_per_day limit.
    """
    # It's good practice to iterate over a copy of keys or items if modifying dict in place,
    # but here we are modifying nested lists, which is generally safer.
    for month_key, month_data in current_chat_history.items():
        if "week" not in month_data:
            continue
        for week_key, week_data in month_data["week"].items():
            if "date" not in week_data:
                continue
            for date_key, daily_turns_list in week_data["date"].items():
                # daily_turns_list is a list of turn objects, e.g., [{"0": [msg1, msg2]}, {"1": [msg3, msg4]}]
                
                new_turns_for_day = []
                messages_counted_for_day = 0
                
                for turn_object in daily_turns_list:
                    if messages_counted_for_day >= max_messages_per_day:
                        break # Already collected enough messages for this day

                    # turn_object is a dictionary with a single key (e.g., "0", "1")
                    # and the value is a list of messages
                    turn_id_key = list(turn_object.keys())[0]
                    messages_in_current_turn = turn_object[turn_id_key]
                    
                    num_messages_in_this_turn = len(messages_in_current_turn)
                    
                    remaining_message_slots = max_messages_per_day - messages_counted_for_day
                    
                    if num_messages_in_this_turn <= remaining_message_slots:
                        # This entire turn can be added
                        new_turns_for_day.append(turn_object)
                        messages_counted_for_day += num_messages_in_this_turn
                    else:
                        # This turn needs to be truncated
                        if remaining_message_slots > 0:
                            truncated_messages = messages_in_current_turn[:remaining_message_slots]
                            new_turns_for_day.append({turn_id_key: truncated_messages})
                            messages_counted_for_day += remaining_message_slots
                        break # Max messages for the day reached
                
                # Replace the old list of turns with the new, trimmed list
                week_data["date"][date_key] = new_turns_for_day
    return current_chat_history # Though modified in place, returning it can be useful

chat_history = trim_chat_history(chat_history)

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

    def main(self):
        self.recall()