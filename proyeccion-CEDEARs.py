import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np

# **Important:** `st.set_page_config` must be the first Streamlit command
st.set_page_config(
  page_title="Proyección de CEDEARs Ajustados por Inflación",
  layout="wide"
)

# Importing the split adjustment function
from inflacion import ajustar_precios_por_splits  # Ensure this module is clean

# Title of the app
st.title("Proyección del rendimiento de CEDEARs ajustados por inflación")

# Load CPI data with caching
@st.cache_data
def load_cpi_data():
  cpi = pd.read_csv('inflaciónargentina2.csv')
  cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
  cpi.set_index('Date', inplace=True)
  # Ensure 'CPI_MoM' is in decimal form (e.g., 5% as 0.05)
  if cpi['CPI_MoM'].max() > 10:
      cpi['CPI_MoM'] = cpi['CPI_MoM'] / 100
  cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
  daily_cpi = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')
  return daily_cpi

# Load CEDEAR conversion ratios with caching
@st.cache_data
def load_cedear_ratios():
  ratios = pd.read_csv('ratioscedears.csv')
  return ratios.set_index('CEDEAR')

# Load data and handle potential errors
try:
  daily_cpi = load_cpi_data()
except FileNotFoundError:
  st.error("El archivo 'inflaciónargentina2.csv' no se encontró.")
  st.stop()
except Exception as e:
  st.error(f"Error al cargar los datos de inflación: {e}")
  st.stop()

try:
  cedear_ratios = load_cedear_ratios()
except FileNotFoundError:
  st.error("El archivo 'ratioscedears.csv' no se encontró.")
  st.stop()
except Exception as e:
  st.error(f"Error al cargar las ratios de CEDEARs: {e}")
  st.stop()

# Sidebar for user inputs
st.sidebar.header("Parámetros de Proyección")

# Inputs for historical and future projections
start_date_default = datetime(2022, 1, 1)
end_date_default = datetime.now() + timedelta(days=365)

start_date = st.sidebar.date_input(
  "Fecha de compra de CEDEARs (Inicio)",
  value=start_date_default,
  min_value=datetime(2000, 1, 1),
  max_value=datetime.now()
)

end_date = st.sidebar.date_input(
  "Fecha de finalización de la predicción",
  value=end_date_default,
  min_value=start_date + timedelta(days=1)
)

if end_date <= start_date:
  st.error("La fecha de finalización debe ser posterior a la fecha de inicio.")
  st.stop()

future_dollar_rate = st.sidebar.number_input(
  "Predicción futura del tipo de cambio (USD/ARS)",
  min_value=0.0,
  value=800.0,
  step=10.0
)

future_inflation_rate = st.sidebar.number_input(
  "Tasa de inflación mensual estimada (%)",
  min_value=0.0,
  max_value=100.0,
  value=5.0,
  step=0.1
) / 100

growth_rate_underlying_asset = st.sidebar.number_input(
  "Tasa de crecimiento anual del activo subyacente (%)",
  min_value=-100.0,
  value=10.0,
  step=0.1
) / 100

# Input for CEDEAR and its underlying asset
ticker = st.sidebar.text_input(
  "Ingresar CEDEAR (por ejemplo, SPY.BA):",
  value="SPY.BA"
).upper()

underlying_ticker = st.sidebar.text_input(
  "Ingresar ticker del activo subyacente (por ejemplo, SPY):",
  value="SPY"
).upper()

# Function to fetch stock data with caching
@st.cache_data(show_spinner=False)
def get_stock_data(ticker, start, end):
  try:
      df = yf.download(ticker, start=start, end=end)
      if df.empty:
          return None
      return df['Adj Close']
  except Exception as e:
      st.error(f"Error al descargar datos para {ticker}: {e}")
      return None

# Fetch stock data for CEDEAR and the underlying asset
with st.spinner("Descargando datos de CEDEAR..."):
  stock_data = get_stock_data(ticker, start_date, end_date)

with st.spinner("Descargando datos del activo subyacente..."):
  underlying_data = get_stock_data(underlying_ticker, start_date, end_date)

# Validate fetched data
if stock_data is None:
  st.error(f"No se encontraron datos para el ticker CEDEAR '{ticker}'. Verifique el símbolo y el rango de fechas.")
  st.stop()

if underlying_data is None:
  st.error(f"No se encontraron datos para el activo subyacente '{underlying_ticker}'. Verifique el símbolo y el rango de fechas.")
  st.stop()

# Adjust stock data for splits
try:
  adjusted_stock_data = ajustar_precios_por_splits(stock_data, ticker)
except Exception as e:
  st.error(f"Error al ajustar los precios por splits: {e}")
  st.stop()

# Retrieve the CEDEAR conversion ratio from the loaded data
# Remove the '.BA' suffix for the lookup
ticker_lookup = ticker.replace('.BA', '')

if ticker_lookup in cedear_ratios.index:
  try:
      conversion_ratio = float(cedear_ratios.loc[ticker_lookup, 'Ratio'])
  except Exception as e:
      st.error(f"Error al obtener el ratio para {ticker_lookup}: {e}")
      st.stop()
else:
  st.error(f"No se encontró un ratio para {ticker_lookup}.")
  st.stop()

# **Corrected Ticker for YPF: Replace 'YPF.BA' with 'YPFD.BA'**
# Fetch the current exchange rate using YPFD.BA and YPF
with st.spinner("Calculando la tasa de cambio actual..."):
  ypf_ba_data = get_stock_data('YPFD.BA', start_date, end_date)  # Corrected Ticker
  ypf_data = get_stock_data('YPF', start_date, end_date)

  if ypf_ba_data is None or ypf_data is None:
      st.error("No se encontraron datos para YPFD.BA o YPF en el rango de fechas seleccionado.")
      st.stop()

  # Ensure there are overlapping dates
  common_dates = ypf_ba_data.index.intersection(ypf_data.index)
  if common_dates.empty:
      st.error("No hay fechas coincidentes entre YPFD.BA y YPF para calcular la tasa de cambio.")
      st.stop()

  # Calculate the current exchange rate
  current_exchange_rate = ypf_ba_data.loc[common_dates].iloc[-1] / ypf_data.loc[common_dates].iloc[-1]

# Future Projections: Prediction of future stock prices and inflation adjustments
projection_start_date = datetime.now()
projection_end_date = end_date
predicted_dates = pd.date_range(start=projection_start_date, end=projection_end_date, freq='D')
num_days = len(predicted_dates)

if num_days == 0:
  st.error("El período de proyección no contiene días.")
  st.stop()

# Initial prices for starting the projection
initial_underlying_price = underlying_data.iloc[-1]  # Last available price of the underlying asset

# Create an array of interpolated exchange rates from the current rate to the future rate
interpolated_exchange_rates = np.linspace(current_exchange_rate, future_dollar_rate, num_days)

# Calculate daily growth rate based on annual growth rate
daily_growth_rate = (1 + growth_rate_underlying_asset) ** (1/365) - 1

# Calculate daily inflation rate based on monthly inflation rate
daily_inflation_rate = (1 + future_inflation_rate) ** (1/30) - 1

# Projection calculations
future_prices_underlying = initial_underlying_price * (1 + daily_growth_rate) ** np.arange(num_days)
projected_stock_prices = (future_prices_underlying / conversion_ratio) * interpolated_exchange_rates
inflation_factors = (1 + daily_inflation_rate) ** np.arange(num_days)
inflation_adjusted_prices = projected_stock_prices / inflation_factors

# Create DataFrame for projected data
projection_df = pd.DataFrame({
  'Date': predicted_dates,
  'Projected Stock Price (ARS)': inflation_adjusted_prices
})

# Compute Projected Cumulative Inflation for Plotting
# Starting from the last historical adjusted price
last_adjusted_price = adjusted_stock_data.iloc[-1]
cumulative_inflation_projected = (1 + daily_inflation_rate) ** np.arange(num_days)
inflation_line_projected = last_adjusted_price * cumulative_inflation_projected

projection_df['Inflación Cumulativa'] = inflation_line_projected

# Plotting the historical and future projections
fig = go.Figure()

# Historical stock price (adjusted for splits)
fig.add_trace(go.Scatter(
  x=adjusted_stock_data.index,
  y=adjusted_stock_data,
  mode='lines',
  name='Precio Histórico (Ajustado por Splits)',
  line=dict(color='blue')
))

# Projected future stock prices (adjusted for inflation)
fig.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['Projected Stock Price (ARS)'],
  mode='lines',
  name='Proyección Ajustada por Inflación',
  line=dict(color='orange', dash='dash')
))

# Projected cumulative inflation
fig.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['Inflación Cumulativa'],
  mode='lines',
  name='Inflación Cumulativa',
  line=dict(color='red', dash='dot')
))

# Layout adjustments
fig.update_layout(
  title="Proyección del rendimiento de CEDEAR ajustado por inflación",
  xaxis_title="Fecha",
  yaxis_title="Precio Ajustado (ARS)",
  legend_title="Leyenda",
  hovermode='x unified'
)

# Display the interactive plot
st.plotly_chart(fig, use_container_width=True)

# Display key metrics
st.subheader("Resumen de Proyección")

final_projected_price = inflation_adjusted_prices[-1]
initial_price = adjusted_stock_data.iloc[-1]

performance = ((final_projected_price / initial_price) - 1) * 100

st.write(f"**Precio inicial ajustado:** ${initial_price:,.2f} ARS")
st.write(f"**Precio proyectado ajustado al {end_date.strftime('%d/%m/%Y')}:** ${final_projected_price:,.2f} ARS")
st.write(f"**Rendimiento proyectado ajustado por inflación:** {performance:.2f}%")
