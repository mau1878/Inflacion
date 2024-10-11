import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import datetime  # Import datetime module

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

# [Existing code for the first part remains unchanged...]

# Big title
st.subheader('2- Ajustadora de acciones del Merval por inflación - MTaurus - https://x.com/MTaurus_ok')

# Option to select the analysis type
analysis_type = st.radio(
  "Selecciona el tipo de análisis que deseas realizar:",
  ('Análisis Individual', 'Comparación de Rendimiento'),
  key='analysis_type_radio'
)

if analysis_type == 'Análisis Individual':
  # [Existing code for individual analysis remains unchanged...]
  pass

else:
  # New code for performance comparison
  st.subheader('Comparación de Rendimiento Ajustado por Inflación')

  # Import datetime and get today's date
  today = datetime.date.today()

  # User input: enter stock tickers (multiple tickers separated by commas)
  tickers_input = st.text_input(
      'Ingresa los tickers de acciones para comparar, separados por comas (por ejemplo, GGAL.BA, METR.BA):',
      key='comparison_tickers_input'
  )

  # User input: select the start date, end date, and zero-percent date
  comparison_start_date = st.date_input(
      'Selecciona la fecha de inicio para el período de comparación:',
      min_value=daily_cpi.index.min().date(),
      max_value=today,
      value=(today - pd.DateOffset(years=1)).date(),
      key='comparison_start_date_input'
  )

  comparison_end_date = st.date_input(
      'Selecciona la fecha de fin para el período de comparación:',
      min_value=comparison_start_date,
      max_value=today,
      value=today,
      key='comparison_end_date_input'
  )

  zero_percent_date = st.date_input(
      'Selecciona la fecha que se considerará como "fecha cero" para el rendimiento:',
      min_value=comparison_start_date,
      max_value=comparison_end_date,
      value=comparison_start_date,
      key='zero_percent_date_input'
  )

  if tickers_input:
      tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
      performance_data = pd.DataFrame()

      for ticker in tickers:
          try:
              # Fetch historical stock data for the comparison period
              stock_data = yf.download(ticker, start=comparison_start_date, end=comparison_end_date + datetime.timedelta(days=1))

              if stock_data.empty:
                  st.error(f"No se encontraron datos para el ticker {ticker}.")
                  continue

              # Ensure the Date column is in datetime format
              stock_data.index = pd.to_datetime(stock_data.index)

              # Adjust stock prices for inflation
              inflation_adjustment = daily_cpi.loc[stock_data.index[-1]] / daily_cpi.loc[stock_data.index]
              stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * inflation_adjustment

              # Ensure the zero_percent_date is within the stock data index
              if pd.to_datetime(zero_percent_date) not in stock_data.index:
                  zero_percent_date_in_data = stock_data.index[0]
                  st.warning(f"La fecha cero '{zero_percent_date}' no está en el rango de datos de {ticker}. Usando {zero_percent_date_in_data.date()} como fecha cero.")
                  base_price = stock_data.iloc[0]['Inflation_Adjusted_Close']
              else:
                  base_price = stock_data.loc[zero_percent_date, 'Inflation_Adjusted_Close']

              # Calculate relative performance based on the zero_percent_date
              stock_data['Relative_Performance'] = (stock_data['Inflation_Adjusted_Close'] / base_price - 1) * 100

              # Add to performance data DataFrame
              performance_data[ticker] = stock_data['Relative_Performance']

          except Exception as e:
              st.error(f"Ocurrió un error con el ticker {ticker}: {e}")

      if not performance_data.empty:
          # Plot the relative performance
          fig = go.Figure()
          for ticker in performance_data.columns:
              fig.add_trace(go.Scatter(
                  x=performance_data.index,
                  y=performance_data[ticker],
                  mode='lines',
                  name=ticker
              ))

          # Add a horizontal line at y=0
          fig.add_shape(
              type='line',
              x0=performance_data.index.min(),
              y0=0,
              x1=performance_data.index.max(),
              y1=0,
              line=dict(color='black', dash='dash')
          )

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
              title='Comparación de Rendimiento Ajustado por Inflación',
              xaxis_title='Fecha',
              yaxis_title='Rendimiento Relativo (%)',
              legend_title_text='Ticker',
              hovermode='x unified'
          )

          st.plotly_chart(fig)
