import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

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
value_choice = st.radio(
    "Do you want to enter the value for the start date or end date?",
    ('Start Date', 'End Date'),
    key='value_choice_radio'
)

if value_choice == 'Start Date':
    start_date = st.date_input(
        'Select the start date:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.min().date(),
        key='start_date_input'
    )
    end_date = st.date_input(
        'Select the end date:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.max().date(),
        key='end_date_input'
    )
    start_value = st.number_input(
        'Enter the value on the start date (in ARS):',
        min_value=0.0,
        value=100.0,
        key='start_value_input'
    )

    # Filter the data for the selected dates
    start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
    end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]

    # Calculate the adjusted value for the end date
    end_value = start_value * (end_inflation / start_inflation)

    # Display the results
    st.write(f"Initial Value on {start_date}: ARS {start_value}")
    st.write(f"Adjusted Value on {end_date}: ARS {end_value:.2f}")

else:
    start_date = st.date_input(
        'Select the start date:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.min().date(),
        key='start_date_end_date_input'
    )
    end_date = st.date_input(
        'Select the end date:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.max().date(),
        key='end_date_end_date_input'
    )
    end_value = st.number_input(
        'Enter the value on the end date (in ARS):',
        min_value=0.0,
        value=100.0,
        key='end_value_input'
    )

    # Filter the data for the selected dates
    start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
    end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]

    # Calculate the adjusted value for the start date
    start_value = end_value / (end_inflation / start_inflation)

    # Display the results
    st.write(f"Adjusted Value on {start_date}: ARS {start_value:.2f}")
    st.write(f"Final Value on {end_date}: ARS {end_value}")

# User input: enter stock tickers (multiple tickers separated by commas)
tickers_input = st.text_input(
    'Enter stock tickers separated by commas (e.g., MSFT, AAPL):',
    key='tickers_input'
)

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    
    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        try:
            # Fetch historical stock data
            stock_data = yf.download(ticker, start=daily_cpi.index.min().date(), end=daily_cpi.index.max().date())
            
            if stock_data.empty:
                st.error(f"No data found for ticker {ticker}.")
                continue

            # Ensure the Date column is in datetime format
            stock_data.index = pd.to_datetime(stock_data.index)

            # Adjust stock prices for inflation
            stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (daily_cpi.loc[stock_data.index[-1]] / daily_cpi.loc[stock_data.index])

            # Plot the adjusted stock prices
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'],
                                     mode='lines', name=ticker))
            
            # If this is the first ticker, calculate and plot the average price as a dotted line
            if i == 0:
                avg_price = stock_data['Inflation_Adjusted_Close'].mean()
                fig.add_trace(go.Scatter(x=stock_data.index, y=[avg_price] * len(stock_data),
                                         mode='lines', name=f'{ticker} Average',
                                         line=dict(dash='dot', color='red')))
        
        except Exception as e:
            st.error(f"An error occurred for ticker {ticker}: {e}")

    # Update layout for the plot
    fig.update_layout(title='Inflation Adjusted Historical Prices',
                      xaxis_title='Date',
                      yaxis_title='Adjusted Close Price (ARS)')

    st.plotly_chart(fig)
