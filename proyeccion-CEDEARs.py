import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
from inflacion import ajustar_precios_por_splits  # Importing split adjustment function

# Load CPI data
@st.cache_data
def load_cpi_data():
    try:
        cpi = pd.read_csv('inflaciónargentina2.csv')
    except FileNotFoundError:
        st.error("El archivo 'inflaciónargentina2.csv' no se encontró.")
        st.stop()

    cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
    cpi.set_index('Date', inplace=True)
    cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
    daily_cpi = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')
    return daily_cpi

# Load CEDEAR conversion ratios
@st.cache_data
def load_cedear_ratios():
    try:
        ratios = pd.read_csv('ratioscedears.csv')
    except FileNotFoundError:
        st.error("El archivo 'ratioscedears.csv' no se encontró.")
        st.stop()
    
    return ratios.set_index('Underlying_Ticker')

daily_cpi = load_cpi_data()
cedear_ratios = load_cedear_ratios()

# User Inputs
st.title("Proyección del rendimiento de CEDEARs ajustados por inflación")

# Inputs for historical and future projections
start_date = st.date_input("Fecha de compra de CEDEARs (Inicio)", value=datetime(2024, 1, 1))
end_date = st.date_input("Fecha de finalización de la predicción", value=datetime.now() + timedelta(days=365))
future_dollar_rate = st.number_input("Predicción futura del tipo de cambio (USD/ARS)", value=800.0)
future_inflation_rate = st.number_input("Tasa de inflación mensual estimada (%)", value=5.0) / 100
growth_rate_underlying_asset = st.number_input("Tasa de crecimiento del activo subyacente (%)", value=10.0) / 100

# Input for CEDEAR and its underlying asset
cedear_ticker = st.text_input("Ingresar CEDEAR (por ejemplo, SPY.BA):", value="SPY.BA")

# Fetch stock data using yfinance
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df['Adj Close']

# Retrieve the underlying asset ticker and conversion ratio from the loaded data
if cedear_ticker in cedear_ratios.index:
    conversion_ratio = cedear_ratios.loc[cedear_ticker, 'Ratio']
    underlying_ticker = cedear_ratios.loc[cedear_ticker, 'Underlying_Ticker']
else:
    st.error(f"No se encontró un ratio para {cedear_ticker}.")
    st.stop()

# Get stock data for both CEDEAR and the underlying asset
stock_data = get_stock_data(cedear_ticker, start_date, end_date)
underlying_data = get_stock_data(underlying_ticker, start_date, end_date)

# Adjust stock data for splits (past data)
adjusted_stock_data = ajustar_precios_por_splits(stock_data, cedear_ticker)

# Future Projections: Prediction of future stock prices and inflation adjustments
predicted_dates = pd.date_range(start=datetime.now(), end=end_date, freq='D')
predicted_stock_prices = []

# Initial prices for starting the projection
initial_stock_price = adjusted_stock_data.iloc[-1]  # Last available stock price (adjusted for splits)
initial_underlying_price = underlying_data.iloc[-1]  # Last available price of the underlying asset

# Project future prices and apply inflation adjustments
for i, date in enumerate(predicted_dates):
    # Calculate the future price of the underlying asset (compounded growth)
    future_price_underlying = initial_underlying_price * (1 + growth_rate_underlying_asset) ** (i / 365)

    # Calculate the future price of the CEDEAR in ARS using the projected dollar rate
    projected_stock_price = (future_price_underlying / conversion_ratio) * future_dollar_rate

    # Apply inflation adjustment based on the monthly inflation rate
    inflation_adjustment = (1 + future_inflation_rate) ** (i / 30)  # Inflation compounded monthly
    inflation_adjusted_price = projected_stock_price / inflation_adjustment

    # Store the future inflation-adjusted stock price
    predicted_stock_prices.append(inflation_adjusted_price)

# Create DataFrame for projected data
projection_df = pd.DataFrame({
    'Date': predicted_dates,
    'Projected Stock Price (ARS)': predicted_stock_prices
})

# Plotting the historical and future projections
fig = go.Figure()

# Historical stock price (adjusted for splits)
fig.add_trace(go.Scatter(
    x=adjusted_stock_data.index,
    y=adjusted_stock_data,
    mode='lines',
    name='Precio Histórico (Ajustado por Splits)'
))

# Projected future stock prices (adjusted for inflation)
fig.add_trace(go.Scatter(
    x=projection_df['Date'],
    y=projection_df['Projected Stock Price (ARS)'],
    mode='lines',
    name='Proyección Ajustada por Inflación'
))

# Layout adjustments
fig.update_layout(
    title="Proyección del rendimiento de CEDEAR ajustado por inflación",
    xaxis_title="Fecha",
    yaxis_title="Precio Ajustado (ARS)",
    legend_title="Leyenda",
)

# Display the interactive plot
st.plotly_chart(fig)
