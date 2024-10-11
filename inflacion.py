import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import logging

# Initialize logger
logger = logging.getLogger(__name__)

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

# Function to adjust stock prices for splits
def ajustar_precios_por_splits(df, ticker):
    try:
        if ticker == 'AGRO.BA':
            # Specific split adjustment for AGRO.BA
            split_date = datetime(2023, 11, 3)
            df.loc[df['Date'] < split_date, 'Close'] /= 6
            df.loc[df['Date'] == split_date, 'Close'] *= 2.1
        else:
            divisor = splits.get(ticker, 1)  # Default divisor is 1 if ticker is not in the dictionary
            split_threshold_date = datetime(2024, 1, 23)
            df.loc[df['Date'] <= split_threshold_date, 'Close'] /= divisor
    except Exception as e:
        logger.error(f"Error adjusting splits for {ticker}: {e}")
    return df

# Function to load CPI data
@st.cache
def load_cpi_data():
    url = 'https://github.com/mau1878/Acciones-del-MERVAL-ajustadas-por-inflaci-n/blob/main/cpi_mom_data.csv?raw=true'
    cpi_data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
    return cpi_data

# Load CPI data
daily_cpi = load_cpi_data()

# Ensure the index of daily_cpi is in datetime.date format
daily_cpi.index = pd.to_datetime(daily_cpi.index).date

# Streamlit app interface
st.title('Ajuste de Precios por Inflación y Splits')

# Input for start and end dates
start_date = st.date_input('Selecciona la fecha de inicio', daily_cpi.index.min())
end_date = st.date_input('Selecciona la fecha de fin', daily_cpi.index.max())

# Convert the input dates to datetime.date format for consistency
start_date = pd.to_datetime(start_date).date()
end_date = pd.to_datetime(end_date).date()

# Input for start value in ARS
start_value = st.number_input('Valor en la fecha de inicio (ARS):', min_value=0.0, step=0.01)

# Ensure that the dates exist in the CPI index
if start_date in daily_cpi.index and end_date in daily_cpi.index:
    start_inflation = daily_cpi.loc[start_date]
    end_inflation = daily_cpi.loc[end_date]

    # Adjust the end value for inflation
    end_value = start_value * (end_inflation / start_inflation)
    st.write(f'El valor ajustado por inflación es: {end_value:.2f} ARS')
else:
    st.error('La fecha seleccionada no está disponible en los datos de inflación.')

# Example data for stocks
ticker = st.selectbox('Seleccionar ticker:', list(splits.keys()))
df = pd.DataFrame({
    'Date': pd.date_range(start='2022-01-01', periods=100),
    'Close': np.random.rand(100) * 1000  # Random stock prices for example
})

# Adjust stock prices for splits
df = ajustar_precios_por_splits(df, ticker)

# Plot the adjusted data
fig = px.line(df, x='Date', y='Close', title=f'Precio ajustado por splits para {ticker}')
st.plotly_chart(fig)
