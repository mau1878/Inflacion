import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
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
# Función para ajustar precios por splits
def ajustar_precios_por_splits(df, ticker):
  try:
      if ticker in splits:
          adjustment = splits[ticker]
          if isinstance(adjustment, tuple):
              # Ajuste con múltiples cambios (por ejemplo, AGRO.BA)
              split_date = datetime(2023, 11, 3)
              df_before_split = df[df.index < split_date].copy()
              df_after_split = df[df.index >= split_date].copy()
              df_before_split['Close'] /= adjustment[0]
              df_after_split['Close'] *= adjustment[1]
              df = pd.concat([df_before_split, df_after_split]).sort_index()
          else:
              # Ajuste simple de split
              split_threshold_date = datetime(2024, 1, 23)
              df.loc[df.index <= split_threshold_date, 'Close'] /= adjustment
      # Si no hay ajuste, no hacer nada
  except Exception as e:
      logger.error(f"Error ajustando splits para {ticker}: {e}")
  return df

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
st.subheader('1- Calculadorita pedorra de precios por inflación. Más abajo la de acciones.')

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
st.subheader('2- Ajustadora de acciones del Merval por inflación - MTaurus - [X: MTaurus_ok](https://x.com/MTaurus_ok)')

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
  value=daily_cpi.index.min().date(),
  key='plot_start_date_input'
)

# Opción para mostrar los valores ajustados por inflación como porcentajes
show_percentage = st.checkbox('Mostrar valores ajustados por inflación como porcentajes', value=False)

# Diccionario para almacenar los datos de acciones procesados
stock_data_dict = {}

if tickers_input:
  tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
  fig = go.Figure()

  # Mapeo de tickers a nombres de variables
  ticker_var_map = {ticker: ticker.replace('.', '_') for ticker in tickers}

  # Descargar y procesar datos nominales (sin ajustar por inflación)
  combined_nominal_df = pd.DataFrame()

  for ticker in tickers:
      try:
          # Descargar datos históricos de la acción desde la fecha seleccionada
          stock_data_nominal = yf.download(ticker, start=plot_start_date, end=daily_cpi.index.max().date())

          if stock_data_nominal.empty:
              st.error(f"No se encontraron datos para el ticker {ticker}.")
              continue

          # Asegurar que el índice sea de tipo datetime
          stock_data_nominal.index = pd.to_datetime(stock_data_nominal.index)

          # Ajustar precios por splits
          stock_data_nominal = ajustar_precios_por_splits(stock_data_nominal, ticker)

          # Almacenar los datos nominales en el DataFrame combinado
          combined_nominal_df[ticker_var_map[ticker]] = stock_data_nominal['Close']

      except Exception as e:
          st.error(f"Ocurrió un error con el ticker {ticker}: {e}")

  # Rellenar datos faltantes
  combined_nominal_df.ffill(inplace=True)
  combined_nominal_df.bfill(inplace=True)

  # Graficar precios nominales
  for i, ticker in enumerate(tickers):
      var_name = ticker_var_map[ticker]
      stock_data_dict[var_name] = combined_nominal_df[var_name]

      if not show_percentage:
          fig.add_trace(go.Scatter(
              x=combined_nominal_df.index,
              y=combined_nominal_df[var_name],
              mode='lines',
              name=f'{ticker} (Nominal)'
          ))

          # Graficar el precio promedio como una línea punteada
          avg_price = combined_nominal_df[var_name].mean()
          fig.add_trace(go.Scatter(
              x=combined_nominal_df.index,
              y=[avg_price] * len(combined_nominal_df),
              mode='lines',
              name=f'{ticker} Precio Promedio',
              line=dict(dash='dot')
          ))

      # Graficar el SMA para el primer ticker solamente
      if i == 0:
          combined_nominal_df[f'{var_name}_SMA'] = combined_nominal_df[var_name].rolling(window=sma_period).mean()
          fig.add_trace(go.Scatter(
              x=combined_nominal_df.index,
              y=combined_nominal_df[f'{var_name}_SMA'],
              mode='lines',
              name=f'{ticker} SMA de {sma_period} Periodos',
              line=dict(color='orange')
          ))

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
          for ticker in sorted_tickers:
              var_name = ticker_var_map[ticker]
              # Usar regex para reemplazar solo ocurrencias completas del ticker
              # Evitar reemplazar partes de otros tickers
              pattern = re.escape(ticker)
              transformed_expression = re.sub(rf'\b{pattern}\b', var_name, transformed_expression)

          # Evaluar la expresión usando el DataFrame combinado nominal
          custom_series_nominal = combined_nominal_df.eval(transformed_expression, engine='python')

          # Ajustar el resultado de la expresión por inflación
          # Unir con los datos de inflación
          custom_series_nominal = custom_series_nominal.to_frame(name='Custom_Nominal')
          custom_series_nominal = custom_series_nominal.join(daily_cpi, how='left')
          # Rellenar hacia adelante cualquier dato de inflación faltante
          custom_series_nominal['Cumulative_Inflation'].ffill(inplace=True)
          # Eliminar cualquier fila restante con NaN en 'Cumulative_Inflation'
          custom_series_nominal.dropna(subset=['Cumulative_Inflation'], inplace=True)
          # Calcular 'Inflation_Adjusted_Custom'
          custom_series_nominal['Inflation_Adjusted_Custom'] = custom_series_nominal['Custom_Nominal'] * (custom_series_nominal['Cumulative_Inflation'].iloc[-1] / custom_series_nominal['Cumulative_Inflation'])

          # Agregar la serie ajustada al gráfico
          if show_percentage:
              # Calcular cambios porcentuales
              custom_series_pct = (custom_series_nominal['Inflation_Adjusted_Custom'] / custom_series_nominal['Inflation_Adjusted_Custom'].iloc[0] - 1) * 100
              fig.add_trace(go.Scatter(
                  x=custom_series_pct.index,
                  y=custom_series_pct,
                  mode='lines',
                  name=f'Custom: {custom_expression} (%)'
              ))

              # Añadir una línea horizontal roja en 0%
              fig.add_shape(
                  type="line",
                  x0=custom_series_pct.index.min(),
                  x1=custom_series_pct.index.max(),
                  y0=0,
                  y1=0,
                  line=dict(color="red", width=2, dash="dash"),
                  xref="x",
                  yref="y"
              )
          else:
              fig.add_trace(go.Scatter(
                  x=custom_series_nominal.index,
                  y=custom_series_nominal['Inflation_Adjusted_Custom'],
                  mode='lines',
                  name=f'Custom: {custom_expression}'
              ))

          # Almacenar la serie ajustada en el diccionario para posibles usos futuros
          stock_data_dict['Custom_Adjusted'] = custom_series_nominal['Inflation_Adjusted_Custom']

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

  # Actualizar el diseño del gráfico
  fig.update_layout(
      title='Precios Históricos Nominales',
      xaxis_title='Fecha',
      yaxis_title='Precio de Cierre (ARS)' if not show_percentage else 'Variación Porcentual (%)',
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
  # Ajustar el precio ajustado por inflación de las acciones individuais para otras visualizaciones si es necesario
  # (Puede ser omitido si no se requiere adicionalmente)

# ------------------------------
# Nueva Sección: Comparación de Volatilidad Ajustada por Inflación y Precio Ajustado por Inflación
st.subheader('4- Comparación de Volatilidad Histórica Ajustada por Inflación y Precio Ajustado por Inflación')

# Entrada del usuario: seleccionar un ticker para la comparación
selected_ticker = st.selectbox(
  'Selecciona una acción para analizar:',
  options=[ticker.upper() for ticker in splits.keys()],
  key='selected_ticker_selectbox'
)

# Entrada del usuario: seleccionar el período de tiempo para la comparación
vol_comparison_start_date = st.date_input(
  'Selecciona la fecha de inicio para la comparación:',
  min_value=daily_cpi.index.min().date(),
  max_value=daily_cpi.index.max().date(),
  value=(daily_cpi.index.max() - pd.Timedelta(days=365)).date(),  # Por defecto, 1 año atrás
  key='vol_comparison_start_date_input'
)
vol_comparison_end_date = st.date_input(
  'Selecciona la fecha de fin para la comparación:',
  min_value=vol_comparison_start_date,
  max_value=daily_cpi.index.max().date(),
  value=daily_cpi.index.max().date(),
  key='vol_comparison_end_date_input'
)

if selected_ticker:
  try:
      # Descargar datos históricos nominales para el ticker seleccionado
      stock_data_nominal = yf.download(selected_ticker, start=vol_comparison_start_date, end=vol_comparison_end_date)

      if stock_data_nominal.empty:
          st.error(f"No se encontraron datos para el ticker {selected_ticker}.")
      else:
          # Ajustar precios por splits
          stock_data_nominal = ajustar_precios_por_splits(stock_data_nominal, selected_ticker)

          # Unir con los datos de inflación
          stock_data_nominal = stock_data_nominal.join(daily_cpi, how='left')
          # Rellenar hacia adelante cualquier dato de inflación faltante
          stock_data_nominal['Cumulative_Inflation'].ffill(inplace=True)
          # Eliminar cualquier fila restante con NaN en 'Cumulative_Inflation'
          stock_data_nominal.dropna(subset=['Cumulative_Inflation'], inplace=True)

          # Calcular Precio Ajustado por Inflación
          stock_data_nominal['Inflation_Adjusted_Close'] = stock_data_nominal['Close'] * (stock_data_nominal['Cumulative_Inflation'].iloc[-1] / stock_data_nominal['Cumulative_Inflation'])

          # Calcular Retornos Diarios Nominales y Ajustados por Inflación
          stock_data_nominal['Return_Nominal'] = stock_data_nominal['Close'].pct_change()
          stock_data_nominal['Return_Adjusted'] = stock_data_nominal['Inflation_Adjusted_Close'].pct_change()

          # Calcular Volatilidad Histórica (Desviación Estándar de los Retornos)
          volatility_nominal = stock_data_nominal['Return_Nominal'].std() * (252**0.5)  # Anualizada
          volatility_adjusted = stock_data_nominal['Return_Adjusted'].std() * (252**0.5)  # Anualizada

          # Mostrar Volatilidades
          st.write(f"**Volatilidad Histórica Nominal:** {volatility_nominal:.2%}")
          st.write(f"**Volatilidad Histórica Ajustada por Inflación:** {volatility_adjusted:.2%}")

          # Graficar Precio Ajustado por Inflación
          fig_vol = go.Figure()

          fig_vol.add_trace(go.Scatter(
              x=stock_data_nominal.index,
              y=stock_data_nominal['Inflation_Adjusted_Close'],
              mode='lines',
              name=f'{selected_ticker} Precio Ajustado por Inflación'
          ))

          # Graficar Volatilidad como una línea horizontal
          fig_vol.add_shape(
              type="line",
              x0=stock_data_nominal.index.min(),
              x1=stock_data_nominal.index.max(),
              y0=volatility_adjusted * stock_data_nominal['Inflation_Adjusted_Close'].mean(),
              y1=volatility_adjusted * stock_data_nominal['Inflation_Adjusted_Close'].mean(),
              line=dict(color="green", width=2, dash="dash"),
              xref="x",
              yref="y",
              name="Volatilidad Ajustada"
          )

          # Actualizar diseño del gráfico
          fig_vol.update_layout(
              title=f'Precio Ajustado por Inflación y Volatilidad Histórica de {selected_ticker}',
              xaxis_title='Fecha',
              yaxis_title='Precio de Cierre Ajustado por Inflación (ARS)',
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
      st.error(f"Ocurrió un error al procesar el ticker {selected_ticker}: {e}")
