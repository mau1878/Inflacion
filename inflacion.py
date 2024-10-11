import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load the inflation data from the CSV file
cpi_data = pd.read_csv('inflaciónargentina2.csv')

# Ensure the Date column is in datetime format with the correct format
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'], format='%d/%m/%Y')

# Set the Date column as the index
cpi_data.set_index('Date', inplace=True)

# Interpolate cumulative inflation directly
cpi_data['Cumulative_Inflation'] = (1 + cpi_data['CPI_MoM']).cumprod()
daily_cpi = cpi_data['Cumulative_Inflation'].resample('D').interpolate(method='linear')

# ------------------------------
# Diccionario de tickers y sus divisores
splits = {
  'MMM.BA': 2,
  'ADGO.BA': 1,
  'ADBE.BA': 2,
  'AEM.BA': 2,
  'AMGN.BA': 3,
  'AAPL.BA': 2,
  'BAC.BA': 2,
  'GOLD.BA': 2,
  'BIOX.BA': 2,
  'CVX.BA': 2,
  'LLY.BA': 7,
  'XOM.BA': 2,
  'FSLR.BA': 6,
  'IBM.BA': 3,
  'JD.BA': 2,
  'JPM.BA': 3,
  'MELI.BA': 2,
  'NFLX.BA': 3,
  'PEP.BA': 3,
  'PFE.BA': 2,
  'PG.BA': 3,
  'RIO.BA': 2,
  'SONY.BA': 2,
  'SBUX.BA': 3,
  'TXR.BA': 2,
  'BA.BA': 4,
  'TM.BA': 3,
  'VZ.BA': 2,
  'VIST.BA': 3,
  'WMT.BA': 3,
  'AGRO.BA': (6, 2.1)  # Ajustes para AGRO.BA
}

# ------------------------------
# Función para ajustar precios por splits
def ajustar_precios_por_splits(df, ticker):
  try:
      if ticker == 'AGRO.BA':
          # Ajuste para AGRO.BA
          split_date = datetime(2023, 11, 3)
          # Dado que el índice es datetime, usamos .loc con fechas
          df.loc[df.index < split_date, 'Close'] /= 6
          df.loc[df.index == split_date, 'Close'] *= 2.1
      else:
          divisor = splits.get(ticker, 1)  # Valor por defecto es 1 si no está en el diccionario
          split_threshold_date = datetime(2024, 1, 23)
          df.loc[df.index <= split_threshold_date, 'Close'] /= divisor
  except Exception as e:
      logger.error(f"Error ajustando splits para {ticker}: {e}")
  return df

# ------------------------------

# Create a Streamlit app
st.title('Ajustadora de acciones del Merval por inflación - MTaurus - https://x.com/MTaurus_ok')

# Subheader for the inflation calculator
st.subheader('1-Calculadorita pedorra de precios por inflación. Más abajo la de acciones.')

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
  try:
      start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
      end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
  except KeyError as e:
      st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
      st.stop()

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
  try:
      start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
      end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
  except KeyError as e:
      st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
      st.stop()

  # Calculate the adjusted value for the start date
  start_value = end_value / (end_inflation / start_inflation)

  # Display the results
  st.write(f"Valor ajustado el {start_date}: ARS {start_value:.2f}")
  st.write(f"Valor final el {end_date}: ARS {end_value}")

# Big title
st.subheader('2- Ajustadora de acciones del Merval por inflación - MTaurus - https://x.com/MTaurus_ok')

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

# Option to show inflation-adjusted values as percentages
show_percentage = st.checkbox('Mostrar valores ajustados por inflación como porcentajes', value=False)

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

          # Adjust stock prices for splits
          stock_data = ajustar_precios_por_splits(stock_data, ticker)

          # Adjust stock prices for inflation
          # Align dates between stock_data and daily_cpi
          stock_data = stock_data.join(daily_cpi, how='left')
          # Forward fill any missing inflation data
          stock_data['Cumulative_Inflation'].ffill(inplace=True)
          # Drop any remaining NaN values
          stock_data.dropna(subset=['Cumulative_Inflation'], inplace=True)
          stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (stock_data['Cumulative_Inflation'].iloc[-1] / stock_data['Cumulative_Inflation'])

          if show_percentage:
              # Calculate inflation-adjusted values as percentages relative to the start value
              stock_data['Inflation_Adjusted_Percentage'] = (stock_data['Inflation_Adjusted_Close'] / stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100
              # Plot the inflation-adjusted percentage changes
              fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Percentage'],
                                       mode='lines', name=f'{ticker} (%)'))

              # Add a red horizontal line at 0% to the plot
              fig.add_shape(
                  type="line",
                  x0=stock_data.index.min(), x1=stock_data.index.max(),
                  y0=0, y1=0,
                  line=dict(color="red", width=2, dash="dash"),
                  xref="x",
                  yref="y"
              )
          else:
              # Plot the adjusted stock prices
              fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'],
                                       mode='lines', name=ticker))

              # Plot the average price as a dotted line (only for absolute values, not percentages)
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
  fig.update_layout(
      title='Precios Históricos Ajustados por Inflación con Promedio y SMA',
      xaxis_title='Fecha',
      yaxis_title='Precio de Cierre Ajustado (ARS)' if not show_percentage else 'Variación Porcentual (%)',
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1
      )
  )

  st.plotly_chart(fig)
