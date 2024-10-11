import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Load CPI data (assuming it has been preloaded or fetched from your file)
cpi_data_url = 'https://github.com/mau1878/Acciones-del-MERVAL-ajustadas-por-inflaci-n/blob/main/cpi_mom_data.csv'
cpi_df = pd.read_csv(cpi_data_url, parse_dates=['Date'], index_col='Date')

# Set up cumulative inflation (from newest to oldest)
cpi_df = cpi_df.sort_index(ascending=False)
daily_cpi = cpi_df['Cumulative_CPI']

# Set up Streamlit inputs
st.title('Análisis de Acciones Ajustadas por Inflación')

# User inputs for selecting tickers, start date, and SMA period
tickers_input = st.text_input('Ingresar manualmente tickers adicionales (separados por comas):')
plot_start_date = st.date_input('Seleccionar fecha de inicio:', value=daily_cpi.index.min().date())
sma_period = st.number_input('SMA (Media Móvil Simple) en días:', min_value=1, value=30)

# Add an option for the user to choose between displaying prices or percentages
display_choice = st.radio(
    "¿Cómo quieres mostrar los datos?",
    ('Precios ajustados por inflación', 'Porcentaje de cambio desde la fecha base')
)

# User input: select the "zero-percent" date for percentage change comparison
zero_percent_date = st.date_input(
    'Selecciona la fecha que servirá como base del 0% (si elegiste mostrar porcentajes):',
    min_value=plot_start_date,
    max_value=daily_cpi.index.max().date(),
    value=daily_cpi.index.min().date(),
    key='zero_percent_date_input'
)

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        try:
            # Fetch historical stock data starting from the user-selected plot start date
            stock_data = yf.download(ticker, start=plot_start_date, end=daily_cpi.index.max().date())
            
            if stock_data.empty:
                st.error(f"No se encontraron datos para el ticker {ticker}.")
                continue

            # Ensure the Date column is in datetime format
            stock_data.index = pd.to_datetime(stock_data.index)

            # Adjust stock prices for inflation
            stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (daily_cpi.loc[stock_data.index[-1]] / daily_cpi.loc[stock_data.index])

            if display_choice == 'Porcentaje de cambio desde la fecha base':
                # Use the zero-percent date to calculate percentage change
                zero_percent_value = stock_data.loc[pd.to_datetime(zero_percent_date), 'Inflation_Adjusted_Close']
                stock_data['Percentage_Change'] = (stock_data['Inflation_Adjusted_Close'] / zero_percent_value - 1) * 100
                y_data = stock_data['Percentage_Change']
                y_label = 'Porcentaje de Cambio (%)'
            else:
                # Show inflation-adjusted prices
                y_data = stock_data['Inflation_Adjusted_Close']
                y_label = 'Precio Ajustado por Inflación (ARS)'

            # Plot the selected data (either prices or percentage change)
            fig.add_trace(go.Scatter(x=stock_data.index, y=y_data,
                                     mode='lines', name=ticker))

            # Plot the SMA for the first ticker only
            if i == 0:
                stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'],
                                         mode='lines', name=f'{ticker} SMA de {sma_period} Periodos',
                                         line=dict(color='orange')))
        
        except Exception as e:
            st.error(f"Ocurrió un error con el ticker {ticker}: {e}")

    # Add a watermark to the plot
    fig.add_annotation(
        text="MTaurus - X: mtaurus_ok",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=30, color="rgba(150, 150, 150, 0.3)"),
        opacity=0.2
    )

    # Update layout for the plot
    fig.update_layout(title='Comparación de rendimiento ajustado por inflación',
                      xaxis_title='Fecha',
                      yaxis_title=y_label)

    # Display the plot
    st.plotly_chart(fig)
