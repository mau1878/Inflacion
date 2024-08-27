import streamlit as st
import pandas as pd

# Load the inflation data from the CSV file
cpi_data = pd.read_csv('inflaciónargentina2.csv')

# Ensure the Date column is in datetime format with the correct format
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'], format='%d/%m/%Y')

# Set the Date column as the index
cpi_data.set_index('Date', inplace=True)

# Resample the data to a daily frequency, using linear interpolation to fill in the gaps
daily_cpi = cpi_data.resample('D').interpolate(method='linear')

# Calculate the cumulative inflation for the daily data
daily_cpi['Cumulative_Inflation'] = (1 + daily_cpi['CPI_MoM']).cumprod()

# Create a Streamlit app
st.title('Inflation Adjustment Calculator')

# User input: initial value and dates
initial_value = st.number_input('Enter the initial value (in ARS):', min_value=0.0, value=100.0)
start_date = st.date_input('Select the start date:', min_value=daily_cpi.index.min().date(), max_value=daily_cpi.index.max().date(), value=daily_cpi.index.min().date())
end_date = st.date_input('Select the end date:', min_value=daily_cpi.index.min().date(), max_value=daily_cpi.index.max().date(), value=daily_cpi.index.max().date())

# Filter the data for the selected dates
start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]['Cumulative_Inflation']
end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]['Cumulative_Inflation']

# Calculate the adjusted value
adjusted_value = initial_value * (end_inflation / start_inflation)

# Display the results
st.write(f"Initial Value on {start_date}: ARS {initial_value}")
st.write(f"Adjusted Value on {end_date}: ARS {adjusted_value:.2f}")
