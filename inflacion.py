import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import logging
import re

# Configurar logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

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
# Function to adjust prices for splits
def ajustar_precios_por_splits(df, ticker):
  try:
      if ticker in splits:
          adjustment = splits[ticker]
          if isinstance(adjustment, tuple):
              # Adjust with multiple changes (e.g., AGRO.BA)
              split_date = datetime(2023, 11, 3)
              df_before_split = df[df.index < split_date].copy()
              df_after_split = df[df.index >= split_date].copy()
              df_before_split['Adj Close'] /= adjustment[0]
              df_after_split['Adj Close'] *= adjustment[1]
              df = pd.concat([df_before_split, df_after_split]).sort_index()
          else:
              # Simple split adjustment
              split_threshold_date = datetime(2024, 1, 23)
              df.loc[df.index <= split_threshold_date, 'Adj Close'] /= adjustment
      # If no adjustment, do nothing
  except Exception as e:
      logger.error(f"Error ajustando splits para {ticker}: {e}")
  return df

# Example of downloading and processing data
ticker = 'METR.BA'
stock_data = yf.download(ticker, start='2000-01-01', end='2024-10-31')

# Flatten the multi-level columns
stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

# Adjust prices for splits
stock_data = ajustar_precios_por_splits(stock_data, ticker)

# ------------------------------
# Cargar datos de inflación desde el archivo CSV
@st.cache_data
def load_cpi_data():
  try:
      cpi = pd.read_csv('inflaciónargentina2.csv')
  except FileNotFoundError:
      st.error("El archivo 'inflaciónargentina2.csv' no se encontró. Asegúrate de que el archivo esté en el mismo directorio que este script.")
      st.stop()
      
  # Asegurar que las columnas necesarias existan
  if 'Date' not in cpi.columns or 'CPI_MoM' not in cpi.columns:
      st.error("El archivo CSV debe contener las columnas 'Date' y 'CPI_MoM'.")
      st.stop()
      
  # Convertir la columna 'Date' a datetime
  cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
  cpi.set_index('Date', inplace=True)
  
  # Calcular la inflación acumulada
  cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
  
  # Resamplear a diario y rellenar
  daily = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')
  return daily

daily_cpi = load_cpi_data()

# ------------------------------
# Crear la aplicación Streamlit
st.title('Ajustadora de acciones del Merval por inflación - MTaurus - [X: MTaurus_ok](https://x.com/MTaurus_ok)')

# Subheader para el calculador de inflación
st.subheader('1- Calculador de precios por inflación')

# Entrada del usuario: elegir si ingresar el valor para la fecha de inicio o fin
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

  # Filtrar los datos para las fechas seleccionadas
  try:
      start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
      end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
  except KeyError as e:
      st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
      st.stop()

  # Calcular el valor ajustado para la fecha de fin
  end_value = start_value * (end_inflation / start_inflation)

  # Mostrar los resultados
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
      min_value=start_date,
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

  # Filtrar los datos para las fechas seleccionadas
  try:
      start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
      end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
  except KeyError as e:
      st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
      st.stop()

  # Calcular el valor ajustado para la fecha de inicio
  start_value = end_value / (end_inflation / start_inflation)

  # Mostrar los resultados
  st.write(f"Valor ajustado el {start_date}: ARS {start_value:.2f}")
  st.write(f"Valor final el {end_date}: ARS {end_value}")

# Subheader para la ajustadora de acciones
st.subheader('2- Ajustadora de acciones por inflación')

# Entrada del usuario: ingresar tickers de acciones (separados por comas)
tickers_input = st.text_input(
  'Ingresa los tickers de acciones separados por comas (por ejemplo, AAPL.BA, MSFT.BA, META):',
  key='tickers_input'
)

# Entrada del usuario: elegir el período de SMA para el primer ticker
sma_period = st.number_input(
  'Ingresa el número de periodos para el SMA del primer ticker:',
  min_value=1,
  value=10,
  key='sma_period_input'
)

# Entrada del usuario: seleccionar la fecha de inicio para los datos mostrados en el gráfico
plot_start_date = st.date_input(
  'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
  min_value=daily_cpi.index.min().date(),
  max_value=daily_cpi.index.max().date(),
  value=(daily_cpi.index.max() - timedelta(days=365)).date(),  # Por defecto, 1 año atrás
  key='plot_start_date_input'
)

# Opción para mostrar los valores ajustados por inflación como porcentajes
show_percentage = st.checkbox('Mostrar valores ajustados por inflación como porcentajes', value=False)

# Add a new checkbox for the alternative percentage calculation
show_percentage_from_recent = st.checkbox('Mostrar valores ajustados por inflación como porcentajes desde el valor más reciente', value=False)

# Diccionario para almacenar los datos de acciones procesados
stock_data_dict_nominal = {}
stock_data_dict_adjusted = {}

if tickers_input:
  tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
  fig = go.Figure()

  # Mapeo de tickers a nombres de variables
  ticker_var_map = {ticker: ticker.replace('.', '_') for ticker in tickers}

  # Descargar y procesar datos para cada ticker
  for i, ticker in enumerate(tickers):
      try:
          # Descargar datos históricos de la acción desde la fecha seleccionada
          stock_data = yf.download(ticker, start=plot_start_date, end=daily_cpi.index.max().date())

          if stock_data.empty:
              st.error(f"No se encontraron datos para el ticker {ticker}.")
              continue

          # Asegurar que el índice sea de tipo datetime
          stock_data.index = pd.to_datetime(stock_data.index)

          # Ajustar precios por splits (solo si está en el diccionario)
          stock_data = ajustar_precios_por_splits(stock_data, ticker)

          # Verificar si el ticker termina con '.BA' para ajustar por inflación
          if ticker.endswith('.BA'):
              # Unir con los datos de inflación
              stock_data = stock_data.join(daily_cpi, how='left')
              # Print the first few rows to verify
              print(stock_data.head())
              # Rellenar hacia adelante cualquier dato de inflación faltante
              stock_data['Cumulative_Inflation'].ffill(inplace=True)
              # Eliminar cualquier fila restante con NaN en 'Cumulative_Inflation'
              stock_data.dropna(subset=['Cumulative_Inflation'], inplace=True)

              # Calcular 'Inflation_Adjusted_Close'
              stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close'] * (stock_data['Cumulative_Inflation'].iloc[-1] / stock_data['Cumulative_Inflation'])
          else:
              # No ajustar por inflación
              stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close']

          # Almacenar los datos en los diccionarios
          var_name = ticker_var_map[ticker]
          stock_data_dict_nominal[var_name] = stock_data['Adj Close']
          stock_data_dict_adjusted[var_name] = stock_data['Inflation_Adjusted_Close']

          if show_percentage or show_percentage_from_recent:
              if show_percentage_from_recent:
                # Calculate the percentage change from each past value to the most recent value
                  stock_data['Inflation_Adjusted_Percentage'] = ((stock_data['Inflation_Adjusted_Close'].iloc[-1] / stock_data['Inflation_Adjusted_Close']) - 1) * 100
                
                # Cap the minimum value at -100%
                  stock_data['Inflation_Adjusted_Percentage'] = stock_data['Inflation_Adjusted_Percentage'].clip(lower=-100)

              else:
                  # Calculate percentages using the initial value as the reference
                  stock_data['Inflation_Adjusted_Percentage'] = (stock_data['Inflation_Adjusted_Close'] / stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100

              # Plot the percentage changes
              fig.add_trace(go.Scatter(
                  x=stock_data.index,
                  y=stock_data['Inflation_Adjusted_Percentage'],
                  mode='lines',
                  name=f'{ticker} (%)',
                  yaxis='y1'
              ))

              # Add a horizontal red line at 0%
              fig.add_shape(
                  type="line",
                  x0=stock_data.index.min(),
                  x1=stock_data.index.max(),
                  y0=0,
                  y1=0,
                  line=dict(color="red", width=2, dash="dash"),
                  xref="x",
                  yref="y1"
              )
          else:
              # Plot the inflation-adjusted prices
              fig.add_trace(go.Scatter(
                  x=stock_data.index,
                  y=stock_data['Inflation_Adjusted_Close'],
                  mode='lines',
                  name=f'{ticker}',
                  yaxis='y1'
              ))

              # Plot the average price as a dotted line
              avg_price = stock_data['Inflation_Adjusted_Close'].mean()
              fig.add_trace(go.Scatter(
                  x=stock_data.index,
                  y=[avg_price] * len(stock_data),
                  mode='lines',
                  name=f'{ticker} Precio Promedio',
                  line=dict(dash='dot'),
                  yaxis='y1'
              ))

          # Graficar el SMA para el primer ticker solamente
          if i == 0:
              stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
              fig.add_trace(go.Scatter(
                  x=stock_data.index,
                  y=stock_data['SMA'],
                  mode='lines',
                  name=f'{ticker} SMA de {sma_period} Periodos',
                  line=dict(color='orange'),
                  yaxis='y1'
              ))

      except Exception as e:
          st.error(f"Ocurrió un error con el ticker {ticker}: {e}")

  # ------------------------------
  # Cálculos o Ratios Personalizados
  st.subheader('3- Cálculos o Ratios Personalizados')

  st.markdown("""
      Puedes definir expresiones matemáticas personalizadas utilizando los tickers cargados.
      **Ejemplo:** `META*(YPFD.BA / YPF)/20`
      
      **Instrucciones:**
      - Usa los tickers tal como los ingresaste (incluyendo `.BA` si corresponde).
      - Los tickers con puntos (`.`) serán automáticamente reemplazados por guiones bajos (`_`) en la evaluación.
      - Por lo tanto, la expresión anterior se transformará internamente a: `META*(YPFD_BA / YPF)/20`
      - Asegúrate de que todos los tickers utilizados en la expresión estén cargados y escritos correctamente.
      - Puedes usar operadores matemáticos básicos: `+`, `-`, `*`, `/`, `**`, etc.
      - Puedes usar funciones de `pandas` como `mean()`, `max()`, etc.
  """)

  custom_expression = st.text_input(
      'Ingresa una expresión personalizada usando los tickers cargados, operadores matemáticos y funciones:',
      placeholder='Por ejemplo: META*(YPFD.BA / YPF)/20',
      key='custom_expression_input'
  )

  if custom_expression:
      try:
          # Reemplazar '.' por '_' en los tickers dentro de la expresión
          # Ordenar tickers por longitud descendente para evitar reemplazos parciales
          sorted_tickers = sorted(ticker_var_map.keys(), key=len, reverse=True)
          transformed_expression = custom_expression
          used_ba_tickers = set()  # Para rastrear si se usa algún .BA ticker

          for ticker in sorted_tickers:
              var_name = ticker_var_map[ticker]
              # Usar regex para reemplazar solo ocurrencias completas del ticker
              pattern = re.escape(ticker)
              transformed_expression = re.sub(rf'\b{pattern}\b', var_name, transformed_expression)
              if ticker.endswith('.BA'):
                  used_ba_tickers.add(ticker)

          # Crear un DataFrame combinado con todas las series nominales
          combined_nominal_df = pd.DataFrame(stock_data_dict_nominal)
          # Rellenar datos faltantes
          combined_nominal_df.ffill(inplace=True)
          combined_nominal_df.bfill(inplace=True)

          # Evaluar la expresión usando el DataFrame combinado nominal
          custom_series_nominal = combined_nominal_df.eval(transformed_expression, engine='python')

          # Crear un DataFrame para la expresión personalizada
          custom_series_nominal = custom_series_nominal.to_frame(name='Custom_Nominal')

          # Determinar si se debe ajustar por inflación
          # Solo ajustar si al menos uno de los tickers usados termina con '.BA'
          if used_ba_tickers:
              # Ajustar por inflación
              custom_series_nominal = custom_series_nominal.join(daily_cpi, how='left')
              # Rellenar hacia adelante cualquier dato de inflación faltante
              custom_series_nominal['Cumulative_Inflation'].ffill(inplace=True)
              # Eliminar cualquier fila restante con NaN en 'Cumulative_Inflation'
              custom_series_nominal.dropna(subset=['Cumulative_Inflation'], inplace=True)
              # Calcular 'Inflation_Adjusted_Custom'
              custom_series_nominal['Inflation_Adjusted_Custom'] = custom_series_nominal['Custom_Nominal'] * (custom_series_nominal['Cumulative_Inflation'].iloc[-1] / custom_series_nominal['Cumulative_Inflation'])
              adjusted_series = custom_series_nominal['Inflation_Adjusted_Custom']
          else:
              # No ajustar por inflación
              adjusted_series = custom_series_nominal['Custom_Nominal']

          # Agregar la serie ajustada al gráfico
          if show_percentage or show_percentage_from_recent:
              if show_percentage_from_recent:
                  # Calculate percentages using the most recent value as the reference
                  custom_series_pct = (adjusted_series / adjusted_series.iloc[-1] - 1) * 100
                  # Invert the sign for past values
                  custom_series_pct = -custom_series_pct
              else:
                  # Calculate percentages using the initial value as the reference
                  custom_series_pct = (adjusted_series / adjusted_series.iloc[0] - 1) * 100

              fig.add_trace(go.Scatter(
                  x=custom_series_pct.index,
                  y=custom_series_pct,
                  mode='lines',
                  name=f'Custom: {custom_expression} (%)',
                  yaxis='y1'
              ))

              # Add a horizontal red line at 0%
              fig.add_shape(
                  type="line",
                  x0=custom_series_pct.index.min(),
                  x1=custom_series_pct.index.max(),
                  y0=0,
                  y1=0,
                  line=dict(color="red", width=2, dash="dash"),
                  xref="x",
                  yref="y1"
              )
          else:
              fig.add_trace(go.Scatter(
                  x=adjusted_series.index,
                  y=adjusted_series,
                  mode='lines',
                  name=f'Custom: {custom_expression}',
                  yaxis='y1'
              ))

          # Almacenar la serie ajustada en el diccionario para posibles usos futuros
          stock_data_dict_adjusted['Custom_Adjusted'] = adjusted_series

      except Exception as e:
          # Obtener nombres de variables disponibles para asistir en la corrección
          available_vars = ', '.join([v for v in ticker_var_map.values()])
          st.error(f"Error al evaluar la expresión personalizada: {e}\n\n**Nombres de variables disponibles:** {available_vars}")

  # ------------------------------

  # Añadir una marca de agua al gráfico
  fig.add_annotation(
      text="MTaurus - X: mtaurus_ok",
      xref="paper", yref="paper",
      x=0.5, y=0.5,
      showarrow=False,
      font=dict(size=30, color="rgba(150, 150, 150, 0.3)"),
      opacity=0.2
  )

  # Configurar los ejes y actualizar el diseño del gráfico
  if show_percentage or show_percentage_from_recent:
      # Solo un eje Y
      fig.update_layout(
          title='Precios Históricos Ajustados por Inflación (%)',
          xaxis_title='Fecha',
          yaxis=dict(
              title='Variación Porcentual (%)',
              titlefont=dict(color='#1f77b4'),
              tickfont=dict(color='#1f77b4')
          ),
          legend=dict(
              orientation="h",
              yanchor="bottom",
              y=1.02,
              xanchor="right",
              x=1
          )
      )
  else:
      # Solo un eje Y
      fig.update_layout(
          title='Precios Históricos Ajustados por Inflación',
          xaxis_title='Fecha',
          yaxis=dict(
              title='Precio de Cierre Ajustado (ARS)',
              titlefont=dict(color='#1f77b4'),
              tickfont=dict(color='#1f77b4')
          ),
          legend=dict(
              orientation="h",
              yanchor="bottom",
              y=1.02,
              xanchor="right",
              x=1
          )
      )

  st.plotly_chart(fig)

# ------------------------------
# Nueva Sección: Comparación de Volatilidad Ajustada por Inflación y Precio Ajustado por Inflación
st.subheader('4- Comparación de Volatilidad Histórica Ajustada por Inflación y Precio Ajustado por Inflación')

# Entrada del usuario: seleccionar un ticker para la comparación
selected_ticker = st.text_input(
  'Ingresa una acción para analizar la volatilidad histórica (puede ser cualquier ticker):',
  placeholder='Por ejemplo: AAPL, AAPL.BA',
  key='selected_ticker_input'
)

# Entrada del usuario: seleccionar el período de tiempo para la comparación
vol_comparison_start_date = st.date_input(
  'Selecciona la fecha de inicio para la comparación:',
  min_value=daily_cpi.index.min().date(),
  max_value=daily_cpi.index.max().date(),
  value=(daily_cpi.index.max() - timedelta(days=365)).date(),  # Por defecto, 1 año atrás
  key='vol_comparison_start_date_input'
)
vol_comparison_end_date = st.date_input(
  'Selecciona la fecha de fin para la comparación:',
  min_value=vol_comparison_start_date,
  max_value=daily_cpi.index.max().date(),
  value=daily_cpi.index.max().date(),
  key='vol_comparison_end_date_input'
)

# Entrada del usuario: seleccionar el número de periodos para calcular la volatilidad histórica
volatility_window = st.number_input(
  'Selecciona el número de periodos para calcular la volatilidad histórica:',
  min_value=1,
  value=20,
  key='volatility_window_input'
)

if selected_ticker:
  ticker = selected_ticker.strip().upper()
  try:
      # Descargar datos históricos para el ticker seleccionado
      stock_data = yf.download(ticker, start=vol_comparison_start_date, end=vol_comparison_end_date)

      if stock_data.empty:
          st.error(f"No se encontraron datos para el ticker {ticker}.")
      else:
          # Ajustar precios por splits si está en el diccionario
          stock_data = ajustar_precios_por_splits(stock_data, ticker)

          # Verificar si el ticker termina con '.BA' para ajustar por inflación
          if ticker.endswith('.BA'):
              # Unir con los datos de inflación
              stock_data = stock_data.join(daily_cpi, how='left')
              # Rellenar hacia adelante cualquier dato de inflación faltante
              stock_data['Cumulative_Inflation'].ffill(inplace=True)
              # Eliminar cualquier fila restante con NaN en 'Cumulative_Inflation'
              stock_data.dropna(subset=['Cumulative_Inflation'], inplace=True)

              # Calcular Precio Ajustado por Inflación
              stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close'] * (stock_data['Cumulative_Inflation'].iloc[-1] / stock_data['Cumulative_Inflation'])
          else:
              # No ajustar por inflación
              stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close']

          # Calcular Retornos Diarios Ajustados por Inflación
          stock_data['Return_Adjusted'] = stock_data['Inflation_Adjusted_Close'].pct_change()

          # Calcular Volatilidad Histórica Ajustada por Inflación sobre la ventana seleccionada
          stock_data['Volatility_Adjusted'] = stock_data['Return_Adjusted'].rolling(window=volatility_window).std() * (252**0.5)  # Anualizada

          # Mostrar Volatilidades
          latest_volatility = stock_data['Volatility_Adjusted'].dropna().iloc[-1]
          st.write(f"**Volatilidad Histórica Ajustada por Inflación (ventana {volatility_window}):** {latest_volatility:.2%}")

          # Crear el gráfico con dos ejes Y
          fig_vol = go.Figure()

          # Trazar Precio Ajustado por Inflación en y1
          fig_vol.add_trace(go.Scatter(
              x=stock_data.index,
              y=stock_data['Inflation_Adjusted_Close'],
              mode='lines',
              name=f'{ticker} Precio Ajustado por Inflación',
              line=dict(color='#1f77b4'),  # Color del eje y1
              yaxis='y1'
          ))

          # Trazar Volatilidad Histórica Ajustada por Inflación en y2
          fig_vol.add_trace(go.Scatter(
              x=stock_data.index,
              y=stock_data['Volatility_Adjusted'],
              mode='lines',
              name=f'{ticker} Volatilidad Histórica Ajustada',
              line=dict(color='#ff7f0e'),  # Color del eje y2
              yaxis='y2'
          ))

          # Añadir una marca de agua al gráfico
          fig_vol.add_annotation(
              text="MTaurus - X: mtaurus_ok",
              xref="paper", yref="paper",
              x=0.5, y=0.5,
              showarrow=False,
              font=dict(size=30, color="rgba(150, 150, 150, 0.3)"),
              opacity=0.2
          )

          # Actualizar el diseño del gráfico para incluir dos ejes Y
          fig_vol.update_layout(
              title=f'Precio Ajustado por Inflación y Volatilidad Histórica de {ticker}',
              xaxis_title='Fecha',
              yaxis=dict(
                  title='Precio de Cierre Ajustado por Inflación (ARS)',
                  titlefont=dict(color='#1f77b4'),
                  tickfont=dict(color='#1f77b4'),
                  anchor='x',
                  side='left'
              ),
              yaxis2=dict(
                  title='Volatilidad Histórica Ajustada (Anualizada)',
                  titlefont=dict(color='#ff7f0e'),
                  tickfont=dict(color='#ff7f0e'),
                  overlaying='y',
                  side='right'
              ),
              legend=dict(
                  orientation="h",
                  yanchor="bottom",
                  y=1.02,
                  xanchor="right",
                  x=1
              )
          )

          st.plotly_chart(fig_vol)

  except Exception as e:
      st.error(f"Ocurrió un error al procesar el ticker {ticker}: {e}")
