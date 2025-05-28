import streamlit as st
import pandas as pd
import utils
import time 


class Agent:
    def __init__(self, user_input, local_var=None):
        self.user_input = user_input
        self.local_var = local_var

    def _user_input(self, user_input):
        if user_input is None:
            effective_user_input = self.user_input
        else:
            effective_user_input = user_input
            
        return effective_user_input 
        
    def json_agent(self, system_prompt, model=None, user_input=None, is_visible=False): 
        # job here is the specific work that the agent will be specified to do 
        # you also have to specify the case where the orchestrator asks the user for clarification

            
        general_system_prompt = """

        You are to output your response in markdown json.

        example: 
        ```json
        {
            
        }
        ```
        """
        general_system_prompt += system_prompt 
        
        real_input = self._user_input(user_input)
        response = utils.get_response(general_system_prompt, real_input, model)
        
        output = utils.display_stream(response, visible=is_visible)

        json = utils.extract_json(output)
        
        return json
        
    def coder(self, system_prompt, model=None, user_input=None, max_tries=3, hide_code=False):
        general_system_prompt = """
        You are a coder agent. 

        Output your code in the following format: 
        ```python
        import ... (required libraries here)
        def main(): 
            ... (actual coding for the problem)
            return ... (return the main variable that contains the information or item you need)
        ```
        """
        general_system_prompt += system_prompt

        real_input = self._user_input(user_input)
        return self._generate_and_execute_code(general_system_prompt, max_tries, model, real_input, hide_code)

    def _generate_and_execute_code(self, system_prompt, remaining_tries, model, user_input, hide_code):
        response = utils.get_response(system_prompt, user_input, model)
        if hide_code:
            output = utils.display_stream(response, visible=False)
        else:
            output = utils.display_stream(response)

        code = utils.extract_python(output)

        # Check if 'features_list_json' exists in session_state AND is not None
        if 'features_list_json' in st.session_state and st.session_state['features_list_json'] is not None:
            try:
                features_array = utils.convert_features_list_to_array(st.session_state['features_list_json'])
                # It seems the original code intended to reassign 'code' here
                code = utils.auto_correct_code_by_list(code, features_array, only_correct_quoted=True) # Assuming only_correct_quoted=True is desired
            except Exception as e:
                st.warning(f"Could not process 'features_list_json' for code auto-correction: {e}")
                # features_array remains an empty list, so auto_correct_code_by_list might be called with an empty list
                # or you might choose to not call it if features_array is empty.
                # If features_array must have content for auto_correct_code_by_list to be useful,
                # you might skip the call if features_array is empty after the try-except.
                if not features_array: # If conversion failed or features_list_json was empty
                    pass # Or log, effectively skipping auto-correction based on features
                else: # If features_array was populated but auto_correct_code_by_list failed (less likely if above try works)
                    code = utils.auto_correct_code_by_list(code, features_array, only_correct_quoted=True)


        result, success = utils.execute_code(code, self.local_var)
        
        if success:
            return result, code
            
        elif remaining_tries > 1:
            system_prompt += """There is something that went wrong with this code. Using the 
    error message below, fix the code and return it. 
    
            First point out what went wrong and how you could fix it. 
            Common mistakes include, but are not limited to:
            1. Trying to access a feature in a dataset that is not there. If that is the error, then look over the list of features, and make sure to only use what is there.
            2. MAKE SURE TO CAPITALIZE THE FEATURE NAMES AND CATEGORIES!!!
            3. Make sure to define the main function and have no parameters in it.
                def main(): 
                    (your code here). 
            
            Often times, look at the intent and try to simplify and change the code to accomplish the task.  
            \n"""
            system_prompt += result
            return self._generate_and_execute_code(system_prompt, remaining_tries - 1, model, user_input, hide_code)
            
        else:
            return result, code
        
    def reporter(self, job, model=None, user_input=None):
        system_prompt = """
        You are part of a multi-agent data analyst system.
        Your job is to take the outputs of the previous agents (usually a coder agent which produces some output) and delegate the findings to user based on the user input. 
        
        """
        system_prompt += job

        real_input = self._user_input(user_input)
        response = utils.get_response(system_prompt, real_input, model)
        output = utils.display_stream(response)

        return output
        
    def chat(self, system_prompt, model=None):
        response = utils.get_response(system_prompt, self.user_input, model)
        content = utils.display_stream(response)

        return content

class ReAct:
    def __init__(self, user_input, functions_and_tools_dict=None, functions_and_tools_description=None): # gives the option for inputting function calling 
        self.user_input = user_input
        self.functions_and_tools_dict = functions_and_tools_dict
        self.functions_and_tools_description = functions_and_tools_description
        self.conversation_thread = {
            0: {
                "reason_output": None,
                "act_work": None,
                "act_output": None
            }
        }
    
    def reason(self, reason_job, act_output=None, iteration=0):
        system_prompt = """
        You are a specialized Reasoning Engine within a ReAct (Reason-Act) multi-agent system. Your core responsibility is to serve as the "brains" of the operation, meticulously planning the next step.

        **Your Task:**

        1.  **Analyze Inputs:** Carefully review the current `user input` and form a step by step plan how to solve the problem. 
        2.  **Formulate a Reasoning Step:** Based on your analysis, determine the most logical and effective next action required to progress towards achieving the specified goals. This involves breaking down complex tasks if necessary.
        3.  **Have a list of the things that the Act agent has discovered thus far, and this should fill up as you learn more information **
        4.  **Generate Actionable Directions:** Craft clear, concise, and unambiguous `directions` for the `Act` agent. These directions should specify a single, concrete action the `Act` agent needs to perform.
        5.  **Determine Task Completion:**
            * If you assess that all goals have been fully achieved and the overall task is complete, set the `finished` flag to `true`.
            * Otherwise, if further actions are required, set the `finished` flag to `false`.

        **Output Format:**

        You MUST output your findings strictly as a JSON object with the following structure:
        
        Think step by step and then give your output. 
        ```json
        {
            "directions": "string // A clear, actionable instruction for the Act agent. Example: 'Extract the key entities from the user's last message.' or 'Query the database for customer_id 75.'",
            "problem_type": "string // One of 2 options, 'python_code', 'function_call'
            "finished": "boolean // true if all goals are met, false otherwise"
        }
        ```

        On your final step when all parts of the problem are done, write true in the 'finished' value. 

        The first initial task will always be to first get to know the data. So if asked to get a particular feature(s) from a table(s), then first see the list of tables, then go from there. 
        """
        system_prompt += f"""Here is the overall job for your agent:  {reason_job}"""

        user_input = f"Iteration: {iteration}\n"
        user_input += self.user_input

        if iteration > 0: 
            user_input += f"Here is the conversation thread thus far: {self.conversation_thread}"
            user_input += f"\nHere is the most recent Act output: {act_output}"
        
        # come back to this eventually want it to direct very complex workflows. it could call 
        if self.functions_and_tools_description is not None:
            user_input += f"Here are a list of tools and functions you have at your disposal {self.functions_and_tools_description}"

        reason_output = Agent(user_input, {}).json_agent(system_prompt, "", user_input, is_visible=True)

        self.conversation_thread[iteration]["reason_output"] = reason_output

        return reason_output
    
    def act(self, reason_output, act_job, iteration=0, problem_type="python_code"):
        system_prompt = f"""
        You are a specialized Act Engine within a ReAct (Reason-Act) multi-agent system. Your primary function is to execute tasks based on precise instructions.

        **Your Task:**

        1.  **Receive Directions:** You will be given `directions` from the Reasoning Engine. These directions will specify a single, concrete action you need to perform.
        2.  **Execute Action:** Carry out the action described in the `directions` as accurately and effectively as possible. This may involve:
            * Accessing information.
            * Utilizing available tools or functions.
            * Performing calculations.
            * Interacting with external systems or APIs (if applicable and configured).
        3.  **Report Outcome:** After attempting the action, you MUST report the outcome. This report is critical for the Reasoning Engine to determine the next steps.

        **Output Format:**

        Your output should be a clear and concise "observation" detailing the result of your action.

        * **If the action is successful:** Describe the result or the data obtained.
            * *Example Direction:* "Extract the user's name from the text: 'Hello, I am David.'"
            * *Example Successful Observation:* "Successfully extracted name: David."
        * **If the action fails or an error occurs:** Clearly state the error or the reason for failure.
            * *Example Direction:* "Query database for order_id 99999."
            * *Example Error Observation:* "Error: Could not connect to the database." or "Error: Order ID 99999 not found."
        * **If the action is to perform a task without a specific data return (e.g., send an email):** Confirm the action was completed.
            * *Example Direction:* "Send a confirmation email to 'user@example.com'."
            * *Example Confirmation Observation:* "Action completed: Confirmation email queued for sending to user@example.com."

        **Key Considerations for Your Actions:**

        * **Precision:** Follow the `directions` exactly as provided.
        * **Clarity in Observation:** Your output (the observation) must be informative enough for the Reasoning Engine to understand the consequence of your action.
        * **Focus:** Perform only the action specified. Do not attempt to reason or plan beyond the given directions.

        Here is your general task and job: {act_job}

        """

        act_input = f"Iteration_{iteration}\t\tHere are the directions given to you from the Reason agent. Follow accordingly."
        act_input += f"\n {reason_output}"
        
        if problem_type == 'python_code':
            result, code = Agent(act_input, {}).coder(system_prompt, "", act_input)

            st.write(result)

            self.conversation_thread[iteration]["act_work"] = code
            self.conversation_thread[iteration]["act_output"] = result 

            act_output = result
        
        return act_output

    def react_loop(self, reason_job, act_job, max_iterations=10):
        current_iteration = 0
        act_output = None

        with st.expander("IPS Algorithm"):
            while current_iteration <= max_iterations:
                # --- FIX FOR KeyError ---
                # Ensure the dictionary structure for the current iteration exists
                if current_iteration not in self.conversation_thread:
                    self.conversation_thread[current_iteration] = {
                        "reason_output": None,
                        "act_work": None,
                        "act_output": None
                    }
                
                reason_output = self.reason(reason_job, act_output, iteration=current_iteration)

                finished = reason_output['finished']
                if finished is True:
                    break

                act_output = self.act(act_job, str(reason_output), iteration=current_iteration)

                current_iteration += 1

        st.write("**Answer**")
        st.write(act_output)

        utils.assistant_message("react_thread", self.conversation_thread)

        return act_output


       