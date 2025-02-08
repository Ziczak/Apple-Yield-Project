import pandas as pd

# Load datasets
ex_weather_df = pd.read_csv('/Users/zhitanwu/DATAPROJECT/DataScience_Project_1/Expanded_ExWeather_PA.csv')
merged_data = pd.read_csv('/Users/zhitanwu/DATAPROJECT/DataScience_Project_1/final_PA.csv')

# Ensure DATE and range columns are in datetime format
key_column = 'DATE'

# Parse dates in ex_weather_df with explicit format for DD/MM/YYYY
ex_weather_df['BEGIN_DATE'] = pd.to_datetime(
    ex_weather_df['BEGIN_DATE'],
    format='%d/%m/%Y',
    errors='coerce'
)
ex_weather_df['END_DATE'] = pd.to_datetime(
    ex_weather_df['END_DATE'],
    format='%d/%m/%Y',
    errors='coerce'
)

# Parse dates in merged_data assuming its original format is DD/MM/YYYY
merged_data[key_column] = pd.to_datetime(
    merged_data[key_column],
    format='%d/%m/%Y',
    errors='coerce'
)

# Filter rows based on damage metrics
filtered_df = ex_weather_df[
    (ex_weather_df['DEATHS_DIRECT'] > 0) |
    (ex_weather_df['INJURIES_DIRECT'] > 0) |
    ((ex_weather_df['DAMAGE_PROPERTY_NUM'] + ex_weather_df['DAMAGE_CROPS_NUM']) > 100000)
]

# Extract unique event types
event_types = filtered_df['EVENT_TYPE'].unique()

# Generate a full date range and create a DataFrame for it
full_date_range = pd.date_range(
    start=merged_data[key_column].min(),
    end=merged_data[key_column].max()
)
date_df = pd.DataFrame(full_date_range, columns=[key_column])

# Initialize one-hot encoding columns for each event type
for event in event_types:
    date_df[event] = 0

# Populate one-hot encoding based on event date ranges
for _, row in filtered_df.iterrows():
    if pd.notnull(row['BEGIN_DATE']) and pd.notnull(row['END_DATE']):
        event_dates = pd.date_range(start=row['BEGIN_DATE'], end=row['END_DATE'])
        date_df.loc[date_df[key_column].isin(event_dates), row['EVENT_TYPE']] = 1

# Merge the one-hot encoded extreme weather data with the merged dataset
final_data = pd.merge(merged_data, date_df, on=key_column, how='left')

# Fill any missing values in event columns with 0
final_data[event_types] = final_data[event_types].fillna(0).astype(int)

# Convert dates back to the DD/MM/YYYY format for output
final_data[key_column] = final_data[key_column].dt.strftime('%d/%m/%Y')

# Save the result
final_data.to_csv('final.csv', index=False)

# Calculate and display the count of each extreme weather event
event_counts = final_data[event_types].sum()
print("Extreme Weather Event Counts:")
print(event_counts)

print("One-hot encoding of extreme weather events completed. File saved as 'final_data_with_encoded_events.csv'.")
