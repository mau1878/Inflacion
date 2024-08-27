import streamlit as st
import pandas as pd

# Load the inflation data from the CSV file
cpi_data = pd.read_csv('inflaciónargentina2.csv')

# Display the column names to help diagnose the issue
st.write("Column names in the uploaded CSV file:", cpi_data.columns)

# Assuming the columns are 'Date' and 'CPI_MoM', continue with the processing
# Ensure the Date column is in datetime format
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])

# Convert MoM CPI rates to cumulative inflation multipliers
cpi_data['Cumulative_Inflation'] = (1 + cpi_data['CPI_MoM']).cumprod()

# Create a Streamlit app
st.title('Inflation Adjustment Calculator')

# User input: initial value and dates
initial_value = st.number_input('Enter the initial value (in ARS):', min_value=0.0, value=100.0)
start_date = st.date_input('Select the start date:', min_value=cpi_data['Date'].min().date(), max_value=cpi_data['Date'].max().date(), value=cpi_data['Date'].min().date())
end_date = st.date_input('Select the end date:', min_value=cpi_data['Date'].min().date(), max_value=cpi_data['Date'].max().date(), value=cpi_data['Date'].max().date())

# Filter the data for the selected dates
start_inflation = cpi_data.loc[cpi_data['Date'] <= pd.to_datetime(start_date)].iloc[-1]['Cumulative_Inflation']
end_inflation = cpi_data.loc[cpi_data['Date'] <= pd.to_datetime(end_date)].iloc[-1]['Cumulative_Inflation']

# Calculate the adjusted value
adjusted_value = initial_value * (end_inflation / start_inflation)

# Display the results
st.write(f"Initial Value on {start_date}: ARS {initial_value}")
st.write(f"Adjusted Value on {end_date}: ARS {adjusted_value:.2f}")
