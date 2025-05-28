import pandas as pd
import numpy as np


file_path = r"C:\Users\O413882\DA_Agent_5_27\DA_Agent_testing_5_27\testing_data\unstrucutred_revenue_data.xlsx"

df = pd.read_excel(file_path)#, sheet_name="2025 Revenue Forecast")
import pandas as pd

# Provided JSON data
data = [
    {'Category': 'Member Months', 'Jan (Actuals)': 4618342, 'Feb (Forecast)': 4614271, 'Mar (Forecast)': 4608531, 'Apr (Forecast)': 4604516, 'May (Forecast)': 4602232, 'Jun (Forecast)': 4597980, 'Jul (Forecast)': 4589843, 'Aug (Forecast)': 4584131, 'Sep (Forecast)': 4580927, 'Oct (Forecast)': 4575576, 'Nov (Forecast)': 4568619, 'Dec (Forecast)': 4564729},
    {'Category': 'Membership Growth', 'Jan (Actuals)': 58253, 'Feb (Forecast)': -4071, 'Mar (Forecast)': -5740, 'Apr (Forecast)': -4015, 'May (Forecast)': -2284, 'Jun (Forecast)': -4252, 'Jul (Forecast)': -8137, 'Aug (Forecast)': -5712, 'Sep (Forecast)': -3204, 'Oct (Forecast)': -5351, 'Nov (Forecast)': -6957, 'Dec (Forecast)': -3890},
    {'Category': 'Monthly Retros', 'Jan (Actuals)': 0, 'Feb (Forecast)': 5229, 'Mar (Forecast)': 350, 'Apr (Forecast)': -2853, 'May (Forecast)': 198, 'Jun (Forecast)': 2545, 'Jul (Forecast)': 2824, 'Aug (Forecast)': -9975, 'Sep (Forecast)': 2166, 'Oct (Forecast)': -9175, 'Nov (Forecast)': 1239, 'Dec (Forecast)': -1637},
    {'Category': 'Member Months with Retros', 'Jan (Actuals)': 4618342, 'Feb (Forecast)': 4619500, 'Mar (Forecast)': 4608881, 'Apr (Forecast)': 4601663, 'May (Forecast)': 4602430, 'Jun (Forecast)': 4600525, 'Jul (Forecast)': 4592667, 'Aug (Forecast)': 4574156, 'Sep (Forecast)': 4583093, 'Oct (Forecast)': 4566401, 'Nov (Forecast)': 4569858, 'Dec (Forecast)': 4563092},
    {'Category': 'YTD Retros', 'Jan (Actuals)': 0, 'Feb (Forecast)': 5229, 'Mar (Forecast)': 5579, 'Apr (Forecast)': 2726, 'May (Forecast)': 2924, 'Jun (Forecast)': 5469, 'Jul (Forecast)': 8293, 'Aug (Forecast)': -1682, 'Sep (Forecast)': 484, 'Oct (Forecast)': -8691, 'Nov (Forecast)': -7452, 'Dec (Forecast)': -9089},
    {'Category': 'YTD Member Months (w/ Retros)', 'Jan (Actuals)': 4618342, 'Feb (Forecast)': 9237842, 'Mar (Forecast)': 13846723, 'Apr (Forecast)': 18448386, 'May (Forecast)': 23050816, 'Jun (Forecast)': 27651341, 'Jul (Forecast)': 32244008, 'Aug (Forecast)': 36818164, 'Sep (Forecast)': 41401257, 'Oct (Forecast)': 45967658, 'Nov (Forecast)': 50537516, 'Dec (Forecast)': 55100608},
    {'Category': 'Dues', 'Jan (Actuals)': 2655765940.6400003, 'Feb (Forecast)': 2655007767.6400003, 'Mar (Forecast)': 2555457660.3555098, 'Apr (Forecast)': 2606621771.3212066, 'May (Forecast)': 2607259600.363716, 'Jun (Forecast)': 2608880285.0450773, 'Jul (Forecast)': 2609400078.4027047, 'Aug (Forecast)': 2590156044.793589, 'Sep (Forecast)': 2603956835.3405423, 'Oct (Forecast)': 2589390562.9508567, 'Nov (Forecast)': 2598366382.0716066, 'Dec (Forecast)': 2606107774.3104086}
]

# 1. Convert the entire list of dictionaries into a single pandas DataFrame
# This already structures the data with 'Category' as a column and months as other columns (features)
print("--- Full Initial DataFrame ---")
df_full = pd.DataFrame(data)
print(df_full)
print("\n" + "="*50 + "\n")

# 2. Define the categories for each logical chunk

# Chunk 1: Membership-related metrics
membership_categories = [
    'Member Months',
    'Membership Growth',
    'Monthly Retros',
    'Member Months with Retros',
    'YTD Retros',
    'YTD Member Months (w/ Retros)'
]

# Chunk 2: Financial metrics (Dues)
financial_categories = [
    'Dues'
]

# 3. Create separate DataFrames for each chunk

# Create DataFrame for membership metrics
# Filter rows based on the 'Category' column and then set 'Category' as the index for a neat format
df_membership_metrics = df_full[df_full['Category'].isin(membership_categories)].set_index('Category')

# Create DataFrame for financial metrics
df_financial_metrics = df_full[df_full['Category'].isin(financial_categories)].set_index('Category')


# 4. Display the chunked DataFrames

print("--- Chunk 1: Membership Metrics ---")
print(df_membership_metrics)
print("\n" + "="*50 + "\n")

print("--- Chunk 2: Financial Metrics (Dues) ---")
print(df_financial_metrics)
print("\n" + "="*50 + "\n")
