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
initial_ce_dear_price = adjusted_stock_data.iloc[-1]  # Last historical adjusted CEDEAR price

# Create an array of interpolated exchange rates from the current rate to the future rate
interpolated_exchange_rates = np.linspace(current_exchange_rate, future_dollar_rate, num_days)

# Calculate daily growth rate based on annual growth rate
daily_growth_rate = (1 + growth_rate_underlying_asset) ** (1/365) - 1

# Calculate daily inflation rate based on monthly inflation rate
daily_inflation_rate = (1 + future_inflation_rate) ** (1/30) - 1

# Generate an array representing each day in the projection period
days_passed = np.arange(num_days)

# **1. Inflation-Following Projection**
# CEDEAR price increases exactly with inflation
inflacion_following_prices = initial_ce_dear_price * (1 + daily_inflation_rate) ** days_passed

# **2. Expected Performance Projection**
# CEDEAR price based on underlying asset growth and future exchange rate
future_prices_underlying = initial_underlying_price * (1 + daily_growth_rate) ** days_passed
expected_performance_prices = (future_prices_underlying / conversion_ratio) * interpolated_exchange_rates

# Create DataFrame for projected data
projection_df = pd.DataFrame({
  'Date': predicted_dates,
  'CEDEAR Sigue Inflación': inflacion_following_prices,
  'CEDEAR Esperado': expected_performance_prices
})

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

# Inflation-Following Projection
fig.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['CEDEAR Sigue Inflación'],
  mode='lines',
  name='CEDEAR Sigue Inflación',
  line=dict(color='green', dash='dash')
))

# Expected Performance Projection
fig.add_trace(go.Scatter(
  x=projection_df['Date'],
  y=projection_df['CEDEAR Esperado'],
  mode='lines',
  name='CEDEAR Esperado',
  line=dict(color='orange', dash='dot')
))

# Optional: Add cumulative inflation for reference
# This can be similar to 'CEDEAR Sigue Inflación', so it's optional
# Uncomment below if you want to include it
# cumulative_inflation_projected = (1 + daily_inflation_rate) ** days_passed
# inflation_line_projected = initial_ce_dear_price * cumulative_inflation_projected
# fig.add_trace(go.Scatter(
#     x=projection_df['Date'],
#     y=inflation_line_projected,
#     mode='lines',
#     name='Inflación Cumulativa',
#     line=dict(color='red', dash='dot')
# ))

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

# Final projected prices
final_inflacion_price = inflacion_following_prices[-1]
final_expected_price = expected_performance_prices[-1]

# Performance calculations
performance_inflacion = ((final_inflacion_price / initial_ce_dear_price) - 1) * 100
performance_expected = ((final_expected_price / initial_ce_dear_price) - 1) * 100

# Display metrics
st.write(f"**Precio inicial ajustado:** ${initial_ce_dear_price:,.2f} ARS")
st.write(f"**Precio proyectado ajustado al {end_date.strftime('%d/%m/%Y')} (Sigue Inflación):** ${final_inflacion_price:,.2f} ARS")
st.write(f"**Rendimiento proyectado ajustado por inflación (Sigue Inflación):** {performance_inflacion:.2f}%")
st.write("---")
st.write(f"**Precio proyectado ajustado al {end_date.strftime('%d/%m/%Y')} (Esperado):** ${final_expected_price:,.2f} ARS")
st.write(f"**Rendimiento proyectado ajustado por inflación (Esperado):** {performance_expected:.2f}%")

# Optional: Display comparison table
st.subheader("Comparación de Proyecciones")

comparison_df = pd.DataFrame({
  'Proyección': ['Sigue Inflación', 'Esperado'],
  'Precio Proyectado (ARS)': [final_inflacion_price, final_expected_price],
  'Rendimiento Proyectado (%)': [performance_inflacion, performance_expected]
})

st.table(comparison_df)
