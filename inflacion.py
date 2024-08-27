import streamlit as st
import pandas as pd

# Load the inflation data from the CSV file
cpi_data = pd.read_csv('inflaciónargentina2.csv')

# Ensure the Date column is in datetime format with the correct format
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'], format='%d/%m/%Y')

# Set the Date column as the index
cpi_data.set_index('Date', inplace=True)

# Interpolate cumulative inflation directly
cpi_data['Cumulative_Inflation'] = (1 + cpi_data['CPI_MoM']).cumprod()
daily_cpi = cpi_data['Cumulative_Inflation'].resample('D').interpolate(method='linear')

# Create a Streamlit app
st.title('Inflation Adjustment Calculator')

# User input: choose to enter the value for the start date or end date
value_choice = st.radio("Do you want to enter the value for the start date or end date?", ('Start Date', 'End Date'))

if value_choice == 'Start Date':
    start_date = st.date_input('Select the start date:', min_value=daily_cpi.index.min().date(), max_value=daily_cpi.index.max().date(), value=daily_cpi.index.min().date())
    end_date = st.date_input('Select the end date:', min_value=daily_cpi.index.min().date(), max_value=daily_cpi.index.max().date(), value=daily_cpi.index.max().date())
    start_value = st.number_input('Enter the value on the start date (in ARS):', min_value=0.0, value=100.0)

    # Filter the data for the selected dates
    start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
    end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]

    # Calculate the adjusted value for the end date
    end_value = start_value * (end_inflation / start_inflation)

    # Display the results
    st.write(f"Initial Value on {start_date}: ARS {start_value}")
    st.write(f"Adjusted Value on {end_date}: ARS {end_value:.2f}")

else:
    start_date = st.date_input('Select the start date:', min_value=daily_cpi.index.min().date(), max_value=daily_cpi.index.max().date(), value=daily_cpi.index.min().date())
    end_date = st.date_input('Select the end date:', min_value=daily_cpi.index.min().date(), max_value=daily_cpi.index.max().date(), value=daily_cpi.index.max().date())
    end_value = st.number_input('Enter the value on the end date (in ARS):', min_value=0.0, value=100.0)

    # Filter the data for the selected dates
    start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
    end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]

    # Calculate the adjusted value for the start date
    start_value = end_value / (end_inflation / start_inflation)

    # Display the results
    st.write(f"Adjusted Value on {start_date}: ARS {start_value:.2f}")
    st.write(f"Final Value on {end_date}: ARS {end_value}")
