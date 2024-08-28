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
st.title('Ajustadora de acciones del Merval por inflación - MTaurus - https://x.com/MTaurus_ok')

# User input: choose to enter the value for the start date or end date
value_choice = st.radio(
    "¿Quieres ingresar el valor para la fecha de inicio o la fecha de fin?",
    ('Fecha de Inicio', 'Fecha de Fin'),
    key='value_choice_radio'
)

if value_choice == 'Fecha de Inicio':
    start_date = st.date_input(
        'Selecciona la fecha de inicio:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.min().date(),
        key='start_date_input'
    )
    end_date = st.date_input(
        'Selecciona la fecha de fin:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.max().date(),
        key='end_date_input'
    )
    start_value = st.number_input(
        'Ingresa el valor en la fecha de inicio (en ARS):',
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
    st.write(f"Valor inicial el {start_date}: ARS {start_value}")
    st.write(f"Valor ajustado el {end_date}: ARS {end_value:.2f}")

else:
    start_date = st.date_input(
        'Selecciona la fecha de inicio:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.min().date(),
        key='start_date_end_date_input'
    )
    end_date = st.date_input(
        'Selecciona la fecha de fin:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.max().date(),
        key='end_date_end_date_input'
    )
    end_value = st.number_input(
        'Ingresa el valor en la fecha de fin (en ARS):',
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
    st.write(f"Valor ajustado el {start_date}: ARS {start_value:.2f}")
    st.write(f"Valor final el {end_date}: ARS {end_value}")

# Big title
st.subheader('Ajustadora de acciones del Merval por inflación - MTaurus - https://x.com/MTaurus_ok')

# User input: enter stock tickers (multiple tickers separated by commas)
tickers_input = st.text_input(
    'Ingresa los tickers de acciones separados por comas (por ejemplo, GGAL.BA, CGPA2.BA):',
    key='tickers_input'
)

# User input: choose the SMA period for the first ticker
sma_period = st.number_input(
    'Ingresa el número de periodos para el SMA del primer ticker:',
    min_value=1,
    value=10,
    key='sma_period_input'
)

# User input: select the start date for the data shown in the plot
plot_start_date = st.date_input(
    'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
    min_value=daily_cpi.index.min().date(),
    max_value=daily_cpi.index.max().date(),
    value=daily_cpi.index.min().date(),
    key='plot_start_date_input'
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

            # Plot the adjusted stock prices
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'],
                                     mode='lines', name=ticker))

            # Plot the average price as a dotted line
            avg_price = stock_data['Inflation_Adjusted_Close'].mean()
            fig.add_trace(go.Scatter(x=stock_data.index, y=[avg_price] * len(stock_data),
                                     mode='lines', name=f'{ticker} Precio Promedio',
                                     line=dict(dash='dot')))

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
    fig.update_layout(title='Precios Históricos Ajustados por Inflación con Promedio y SMA',
                      xaxis_title='Fecha',
                      yaxis_title='Precio de Cierre Ajustado (ARS)')

    st.plotly_chart(fig)
