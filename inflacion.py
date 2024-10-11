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

# Big title
st.subheader('Comparación de rendimiento ajustado por inflación')

# User input: enter stock tickers (multiple tickers separated by commas)
tickers_input = st.text_input(
    'Ingresa los tickers de acciones separados por comas (por ejemplo, GGAL.BA, CGPA2.BA):',
    key='tickers_input'
)

# User input: select the start date for the data shown in the plot
plot_start_date = st.date_input(
    'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
    min_value=daily_cpi.index.min().date(),
    max_value=daily_cpi.index.max().date(),
    value=daily_cpi.index.min().date(),
    key='plot_start_date_input'
)

# User input: select the "zero-percent date" to serve as a reference point
zero_percent_date = st.date_input(
    'Selecciona la fecha que será considerada como referencia de 0% (fecha base):',
    min_value=daily_cpi.index.min().date(),
    max_value=daily_cpi.index.max().date(),
    value=daily_cpi.index.max().date(),
    key='zero_percent_date_input'
)

# User input: toggle between showing prices or percentage changes
show_as_percentage = st.checkbox('Mostrar el rendimiento como porcentaje en vez de precios ajustados')

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

    fig = go.Figure()

    for ticker in tickers:
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

            # Calculate the price on the zero-percent date for reference
            zero_percent_price = stock_data.loc[zero_percent_date, 'Inflation_Adjusted_Close']

            if show_as_percentage:
                # Calculate the percentage change relative to the zero-percent date
                stock_data['Percentage_Change'] = (stock_data['Inflation_Adjusted_Close'] / zero_percent_price - 1) * 100
                y_values = stock_data['Percentage_Change']
                y_label = 'Cambio porcentual respecto a la fecha base (%)'
            else:
                # Use the inflation-adjusted prices directly
                y_values = stock_data['Inflation_Adjusted_Close']
                y_label = 'Precio de Cierre Ajustado (ARS)'

            # Plot the adjusted stock prices or percentage changes
            fig.add_trace(go.Scatter(x=stock_data.index, y=y_values,
                                     mode='lines', name=ticker))

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

    st.plotly_chart(fig)
