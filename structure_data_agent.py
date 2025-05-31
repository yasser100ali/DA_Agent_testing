import utils
from agents import ReAct

class FixData:
    def __init__(self, df):
        self.df = df
    
    def structure_data(self):
        system_prompt = """
        You are an ReAct Style AI Agent with a very important task. 
        Your job is to take unstructured data and structure it. 

        By unstructured I mean that the columns are not neatly at the top and the features may not even be placed vertically as columns but as rows. 

        Structure the data so that it is in a clean format similar to how a data sheet would look like when pulled from SQL (all the features are the top row, the data is cleanly organized and placed properly). 
        
        

        """

        agent = ReAct()