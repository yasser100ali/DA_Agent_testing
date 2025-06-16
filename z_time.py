import re
from datetime import datetime, timedelta

def sum_time_ranges(time_ranges_string):
    """
    Calculates the total duration from a string of time ranges.

    Args:
        time_ranges_string: A string containing one or more time ranges,
                            each on a new line, in the format "H:MMam/pm - H:MMam/pm".
                            Example:
                            "8:40am - 12:28pm\n1:02pm - 5:01pm\n6:17pm - 9:41pm"

    Returns:
        A string representing the total duration in hours and minutes,
        e.g., "X hours and Y minutes".
        Returns an error message if the input format is invalid.
    """
    total_duration = timedelta()
    lines = time_ranges_string.strip().split('\n')

    if not time_ranges_string.strip():
        return "0 hours and 0 minutes"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Corrected the regex to use a hyphen-minus '-' instead of an en-dash '–'
        match = re.fullmatch(r"(\d{1,2}:\d{2}(?:am|pm))\s*-\s*(\d{1,2}:\d{2}(?:am|pm))", line, re.IGNORECASE)
        if not match:
            # Also update the expected format in the error message for clarity
            return f"Error: Invalid format in line: '{line}'. Expected 'H:MMam/pm - H:MMam/pm'."

        start_time_str, end_time_str = match.groups()

        try:
            start_time = datetime.strptime(start_time_str, "%I:%M%p")
            end_time = datetime.strptime(end_time_str, "%I:%M%p")
        except ValueError:
            return f"Error: Invalid time value in line: '{line}'."

        if end_time < start_time:
            # This usually implies crossing midnight. For summing segments within a day,
            # this could be an issue or require adding 24 hours to end_time.
            # For this specific problem, we assume logical order or same-day ranges.
            # If your use case requires handling overnight ranges (e.g., 10:00pm - 2:00am),
            # you would add a day to end_time here:
            # end_time += timedelta(days=1)
            # However, for the provided example, this isn't necessary and might indicate an input error.
            # We'll proceed assuming valid intra-day ranges.
            pass


        duration = end_time - start_time
        total_duration += duration

    total_seconds = total_duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return f"{hours} hours and {minutes} minutes"

time_data = """
8:35am - 12:56pm
1:42pm - 5:38pm
7:06pm - 9:52pm
"""

result = sum_time_ranges(time_data)
print(result)