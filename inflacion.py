import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime

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

# Función para ajustar precios por splits
def ajustar_precios_por_splits(df, ticker):
    try:
        if ticker == 'AGRO.BA':
            # Ajuste especial para AGRO.BA
            split_date = datetime(2023, 11, 3)
            df.loc[df.index < split_date, 'Close'] /= 6
            df.loc[df.index == split_date, 'Close'] *= 2.1
        else:
            divisor = splits.get(ticker, 1)  # Valor por defecto es 1 si no está en el diccionario
            split_threshold_date = datetime(2024, 1, 23)
            df.loc[df.index <= split_threshold_date, 'Close'] /= divisor
    except Exception as e:
        st.error(f"Error ajustando splits para {ticker}: {e}")
    return df

# Cargar los datos de inflación
cpi_data = pd.read_csv('inflaciónargentina2.csv')
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'], format='%d/%m/%Y')
cpi_data.set_index('Date', inplace=True)
cpi_data['Cumulative_Inflation'] = (1 + cpi_data['CPI_MoM']).cumprod()
daily_cpi = cpi_data['Cumulative_Inflation'].resample('D').interpolate(method='linear')

# Streamlit App
st.title('Ajustadora de acciones del Merval por inflación - MTaurus')

st.subheader('1 - Calculadora de Precios Ajustados por Inflación')

value_choice = st.radio(
    "¿Quieres ingresar el valor para la fecha de inicio o la fecha de fin?",
    ('Fecha de Inicio', 'Fecha de Fin')
)

if value_choice == 'Fecha de Inicio':
    start_date = st.date_input('Selecciona la fecha de inicio', daily_cpi.index.min().date())
    end_date = st.date_input('Selecciona la fecha de fin', daily_cpi.index.max().date())
    start_value = st.number_input('Valor en la fecha de inicio (ARS):', value=100.0)

    start_inflation = daily_cpi.loc[start_date]
    end_inflation = daily_cpi.loc[end_date]
    end_value = start_value * (end_inflation / start_inflation)

    st.write(f"Valor inicial: {start_value} ARS")
    st.write(f"Valor ajustado al {end_date}: {end_value:.2f} ARS")

else:
    start_date = st.date_input('Fecha de inicio', daily_cpi.index.min().date())
    end_date = st.date_input('Fecha de fin', daily_cpi.index.max().date())
    end_value = st.number_input('Valor en la fecha de fin (ARS):', value=100.0)

    start_inflation = daily_cpi.loc[start_date]
    end_inflation = daily_cpi.loc[end_date]
    start_value = end_value / (end_inflation / start_inflation)

    st.write(f"Valor ajustado al {start_date}: {start_value:.2f} ARS")
    st.write(f"Valor final: {end_value} ARS")

st.subheader('2 - Ajustadora de acciones del Merval')

tickers_input = st.text_input('Ingresa tickers de acciones (separados por comas)', 'GGAL.BA, CGPA2.BA')
sma_period = st.number_input('Número de periodos para el SMA del primer ticker', min_value=1, value=10)
plot_start_date = st.date_input('Fecha de inicio para datos del gráfico', daily_cpi.index.min().date())
show_percentage = st.checkbox('Mostrar valores ajustados como porcentajes')

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        try:
            stock_data = yf.download(ticker, start=plot_start_date, end=daily_cpi.index.max().date())

            if stock_data.empty:
                st.error(f"No se encontraron datos para {ticker}.")
                continue

            stock_data.index = pd.to_datetime(stock_data.index)
            stock_data = ajustar_precios_por_splits(stock_data, ticker)

            stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (daily_cpi.loc[stock_data.index[-1]] / daily_cpi.loc[stock_data.index])

            if show_percentage:
                stock_data['Inflation_Adjusted_Percentage'] = (stock_data['Inflation_Adjusted_Close'] / stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Percentage'],
                                         mode='lines', name=f'{ticker} (%)'))
                fig.add_shape(type="line", x0=stock_data.index.min(), x1=stock_data.index.max(), y0=0, y1=0,
                              line=dict(color="red", width=2, dash="dash"), name='Cero Porcentaje')
            else:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'],
                                         mode='lines', name=ticker))

            if i == 0:
                stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'],
                                         mode='lines', name=f'{ticker} SMA de {sma_period} Periodos', line=dict(color='orange')))

        except Exception as e:
            st.error(f"Ocurrió un error con el ticker {ticker}: {e}")

    fig.update_layout(title='Precios Históricos Ajustados por Inflación y Splits',
                      xaxis_title='Fecha',
                      yaxis_title='Precio de Cierre Ajustado (ARS)' if not show_percentage else 'Variación Porcentual (%)')

    st.plotly_chart(fig)
