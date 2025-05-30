dictionary = {
    "step1": {
        "agent": "base",
        "instructions": "testing base "
    },
    "step2": {
        "agent": "deepinsights",
        "instructions": "testing deepinsights"
    },
    "step3": {
        "agent": "sql",
        "instructions": "pull the following data..."
    }
}



for step in dictionary.values():
    agent = step["agent"]
    inst = step["instructions"] 

    params = {
        "user_input": inst
    }

    if agent in ["base", "deepinsights"]:
        params["local_var"] = "local_var"

    print("params for ", agent)
    print(params)
