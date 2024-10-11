import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

# Diccionario de tickers y sus divisores (ajustes por splits)
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

# Función para ajustar precios por splits
def ajustar_precios_por_splits(df, ticker):
    try:
        if ticker == 'AGRO.BA':
            # Ajuste para AGRO.BA
            split_date = datetime(2023, 11, 3)
            df.loc[df['Date'] < split_date, 'Close'] /= 6
            df.loc[df['Date'] == split_date, 'Close'] *= 2.1
        else:
            divisor = splits.get(ticker, 1)  # Valor por defecto es 1 si no está en el diccionario
            split_threshold_date = datetime(2024, 1, 23)
            df.loc[df['Date'] <= split_threshold_date, 'Close'] /= divisor
    except Exception as e:
        st.error(f"Error ajustando splits para {ticker}: {e}")
    return df

# Función para ajustar los precios por inflación
def ajustar_por_inflacion(df, cpi_df):
    df['Date'] = pd.to_datetime(df['Date'])
    cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])
    
    df = df.merge(cpi_df[['Date', 'Cumulative_CPI']], on='Date', how='left')
    
    # Llenar valores de inflación faltantes con el valor anterior disponible
    df['Cumulative_CPI'].fillna(method='ffill', inplace=True)
    
    # Ajustar precios por la inflación
    df['Close_ajustado'] = df['Close'] / df['Cumulative_CPI']
    
    return df

# Cargar datos de ejemplo de precios e inflación (esto sería cargado desde archivos reales)
@st.cache_data
def cargar_datos():
    daily_cpi = pd.read_csv("https://raw.githubusercontent.com/mau1878/Acciones-del-MERVAL-ajustadas-por-inflaci-n/main/cpi_mom_data.csv")
    daily_cpi['Date'] = pd.to_datetime(daily_cpi['Date'])
    return daily_cpi

# Función principal del script
def main():
    st.title("Ajuste de Precios de Acciones por Inflación y Splits")
    
    # Cargar datos de inflación
    daily_cpi = cargar_datos()

    # Selección de fechas
    start_date = st.date_input('Selecciona la fecha de inicio', daily_cpi['Date'].min())
    end_date = st.date_input('Selecciona la fecha de fin', daily_cpi['Date'].max())
    start_value = st.number_input('Valor en la fecha de inicio (ARS):', min_value=0.0, value=100.0)

    # Ajuste por inflación
    try:
        start_inflation = daily_cpi.loc[daily_cpi['Date'] == pd.to_datetime(start_date), 'Cumulative_CPI'].values[0]
        end_inflation = daily_cpi.loc[daily_cpi['Date'] == pd.to_datetime(end_date), 'Cumulative_CPI'].values[0]
        
        # Calcular el valor ajustado por inflación
        end_value = start_value * (end_inflation / start_inflation)
        st.write(f'El valor ajustado por inflación es: {end_value:.2f} ARS')
    
    except IndexError:
        st.error('La fecha seleccionada no está disponible en los datos de inflación.')
    
    # Cargar precios de acciones
    ticker = st.text_input('Ingresa el ticker del activo:', value='AAPL.BA').upper()

    if ticker in splits:
        # Supongamos que tenemos un DataFrame con datos históricos de precios
        # Aquí se cargarían datos reales de un archivo CSV o API como yfinance
        df = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
            'Close': np.random.uniform(100, 200, size=100)  # Valores ficticios
        })
        
        # Ajustar por splits
        df = ajustar_precios_por_splits(df, ticker)
        
        # Ajustar por inflación
        df_ajustado = ajustar_por_inflacion(df, daily_cpi)
        
        # Graficar
        fig = px.line(df_ajustado, x='Date', y='Close_ajustado', title=f'Precio ajustado de {ticker} por inflación y splits')
        st.plotly_chart(fig)
    else:
        st.error(f'El ticker {ticker} no se encuentra en la lista de ajustes por splits.')

if __name__ == '__main__':
    main()
