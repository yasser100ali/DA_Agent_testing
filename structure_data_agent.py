import utils
from agents import ReAct

class FixData:
    def __init__(self, df):
        self.df = {"df": df}

    def structure_data(self):
        reason_job = """
        You are an ReAct Style AI Agent with a very important task. 
        Your job is to take unstructured data and structure it. 

        By unstructured I mean that the columns are not neatly at the top and the features may not even be placed vertically as columns but as rows. 

        Structure the data so that it is in a clean format similar to how a data sheet would look like when pulled from SQL (all the features are the top row, the data is cleanly organized and placed properly). 
        
        Start by looking over the file, try to see as much of the file as you can (usually not that large so you could) and gradually get more specific on the various key metrics and features.

        Your final output output should be the original unstructured data with all the same information organized so that all the main features are at the top. 

        If there is a data column or time variable anywhere and is structured horizontally change this so that it becomes the first feature in the first column. 

        You have a total of 8 iterations to solve your problem. Make sure to wrap up by iteration 8. 

        Make sure you don't have duplicate dolumn names otherwise it will not pass and make sure there are no unnamed columns. Format it according to your best judgment. 

        """

        act_job = f"""
        You are a python agent, specifically specializing in learning the structure of a given dataframe and structuring it by following a Reason agent instructions. 

        You are part of a ReAct loop and will follow directions according to Reason's dictates.
        when connecting to the database, the path is 

        Return your output as a JSON where you write the important details of various aspects of what you found. 

        1. Write complete self-contained code that includes all necessary imports
        2. You are given file 'df', and it is already a pandas dataframe, simply call it in your main() function. 

        Make sure you don't have duplicate dolumn names otherwise it will not pass and make sure there are no unnamed columns. Format it according to your best judgment. 

        """

        
        react_agent = ReAct("", local_var=self.df)
        final_output = react_agent.react_loop(reason_job, act_job, max_iterations=8)

        return final_output

    
