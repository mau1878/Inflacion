import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Cargar datos de inflación desde el archivo CSV
@st.cache_data
def load_cpi_data():
  cpi = pd.read_csv('inflaciónargentina2.csv')
  cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
  cpi.set_index('Date', inplace=True)
  cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
  daily = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')
  return daily

daily_cpi = load_cpi_data()

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
      if ticker == 'AGRO.BA' and isinstance(splits[ticker], tuple):
          # Ajuste para AGRO.BA con múltiples ajustes
          split_date = datetime(2023, 11, 3)
          df_before_split = df[df.index < split_date].copy()
          df_after_split = df[df.index >= split_date].copy()
          df_before_split['Close'] /= splits[ticker][0]
          df_after_split['Close'] *= splits[ticker][1]
          df = pd.concat([df_before_split, df_after_split]).sort_index()
      else:
          divisor = splits.get(ticker, 1)  # Valor por defecto es 1 si no está en el diccionario
          split_threshold_date = datetime(2024, 1, 23)
          df.loc[df.index <= split_threshold_date, 'Close'] /= divisor
  except Exception as e:
      logger.error(f"Error ajustando splits para {ticker}: {e}")
  return df

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
  'Ingresa los tickers de acciones separados por comas (por ejemplo, GGAL.BA, CGPA2.BA):',
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

  for i, ticker in enumerate(tickers):
      try:
          # Descargar datos históricos de la acción desde la fecha seleccionada
          stock_data = yf.download(ticker, start=plot_start_date, end=daily_cpi.index.max().date())

          if stock_data.empty:
              st.error(f"No se encontraron datos para el ticker {ticker}.")
              continue

          # Asegurar que el índice sea de tipo datetime
          stock_data.index = pd.to_datetime(stock_data.index)

          # Ajustar precios por splits
          stock_data = ajustar_precios_por_splits(stock_data, ticker)

          # Ajustar precios por inflación
          # Alinear fechas entre stock_data y daily_cpi
          stock_data = stock_data.join(daily_cpi, how='left')
          # Rellenar hacia adelante cualquier dato de inflación faltante
          stock_data['Cumulative_Inflation'].ffill(inplace=True)
          # Eliminar cualquier fila restante con NaN en 'Cumulative_Inflation'
          stock_data.dropna(subset=['Cumulative_Inflation'], inplace=True)
          # Calcular 'Inflation_Adjusted_Close'
          stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (stock_data['Cumulative_Inflation'].iloc[-1] / stock_data['Cumulative_Inflation'])

          # Almacenar los datos procesados en el diccionario
          stock_data_dict[ticker] = stock_data.copy()

          if show_percentage:
              # Calcular valores ajustados por inflación como porcentajes relativos al valor inicial
              stock_data['Inflation_Adjusted_Percentage'] = (stock_data['Inflation_Adjusted_Close'] / stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100
              # Graficar los cambios porcentuales ajustados por inflación
              fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Percentage'],
                                       mode='lines', name=f'{ticker} (%)'))

              # Añadir una línea horizontal roja en 0%
              fig.add_shape(
                  type="line",
                  x0=stock_data.index.min(), x1=stock_data.index.max(),
                  y0=0, y1=0,
                  line=dict(color="red", width=2, dash="dash"),
                  xref="x",
                  yref="y"
              )
          else:
              # Graficar los precios ajustados por inflación
              fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'],
                                       mode='lines', name=ticker))

              # Graficar el precio promedio como una línea punteada
              avg_price = stock_data['Inflation_Adjusted_Close'].mean()
              fig.add_trace(go.Scatter(x=stock_data.index, y=[avg_price] * len(stock_data),
                                       mode='lines', name=f'{ticker} Precio Promedio',
                                       line=dict(dash='dot')))

          # Graficar el SMA para el primer ticker solamente
          if i == 0:
              stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
              fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'],
                                       mode='lines', name=f'{ticker} SMA de {sma_period} Periodos',
                                       line=dict(color='orange')))

      except Exception as e:
          st.error(f"Ocurrió un error con el ticker {ticker}: {e}")

  # ------------------------------
  # Cálculos o Ratios Personalizados
  st.subheader('3- Cálculos o Ratios Personalizados')

  st.markdown("""
      Puedes definir expresiones matemáticas personalizadas utilizando los tickers cargados.
      **Ejemplo:** `(META.BA * (YPFD.BA / YPF.BA)) / 20`
      
      **Nombres de Variables Disponibles:**
      - Reemplaza los puntos (`.`) por guiones bajos (`_`).
      - Por ejemplo, `META.BA` se convierte en `META_BA`, `YPFD.BA` en `YPFD_BA`, etc.
  """)

  custom_expression = st.text_input(
      'Ingresa una expresión personalizada usando los tickers cargados, operadores matemáticos y funciones:',
      placeholder='Por ejemplo: (META.BA * (YPFD.BA / YPF.BA)) / 20',
      key='custom_expression_input'
  )

  if custom_expression:
      try:
          # Preparar el diccionario local con las Series de los tickers (reemplazando '.' por '_')
          local_dict = {ticker.replace('.', '_'): data['Inflation_Adjusted_Close'] for ticker, data in stock_data_dict.items()}

          # Reemplazar los nombres de tickers en la expresión con los nombres de variables válidos
          expression = custom_expression
          for ticker in stock_data_dict.keys():
              expression = expression.replace(ticker, ticker.replace('.', '_'))

          # Evaluar la expresión usando pandas.eval
          custom_series = pd.eval(expression, local_dict=local_dict)

          # Nombre para el cálculo personalizado
          custom_name = f'Custom: {custom_expression}'

          # Graficar la serie personalizada
          if show_percentage:
              # Si se muestran porcentajes, calcular el cambio porcentual relativo al inicio
              custom_series_pct = (custom_series / custom_series.iloc[0] - 1) * 100
              fig.add_trace(go.Scatter(x=custom_series_pct.index, y=custom_series_pct,
                                       mode='lines', name=custom_name))
              # Añadir una línea horizontal en 0% si no está presente
              fig.add_shape(
                  type="line",
                  x0=custom_series_pct.index.min(), x1=custom_series_pct.index.max(),
                  y0=0, y1=0,
                  line=dict(color="red", width=2, dash="dash"),
                  xref="x",
                  yref="y"
              )
          else:
              fig.add_trace(go.Scatter(x=custom_series.index, y=custom_series,
                                       mode='lines', name=custom_name))

      except Exception as e:
          # Obtener nombres de variables disponibles para asistir en la corrección
          available_vars = ', '.join(local_dict.keys())
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
