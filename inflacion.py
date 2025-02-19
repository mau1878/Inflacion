import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import logging
import re
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurar logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ------------------------------
# Diccionario de tickers y sus divisores
splits = {
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
    'AGRO.BA': (6, 2.1)
}

# ------------------------------
# Data source functions
def descargar_datos_yfinance(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        if not stock_data.empty:
            # Extract just the Adj Close column and flatten the MultiIndex
            if isinstance(stock_data.columns, pd.MultiIndex):
                adj_close = stock_data['Close'][ticker].to_frame('Adj Close')
            else:
                adj_close = stock_data[['Close']]
            return adj_close
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error downloading data from yfinance for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'ChyrpSession': '0e2b2109d60de6da45154b542afb5768',
            'i18next': 'es',
            'PHPSESSID': '5b8da4e0d96ab5149f4973232931f033',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'dnt': '1',
            'referer': 'https://analisistecnico.com.ar/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://analisistecnico.com.ar/services/datafeed/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Adj Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            logger.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error downloading data from analisistecnico for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_iol(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'intencionApertura': '0',
            '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1',
            'isLogged': '1',
            'uid': '1107644',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'referer': 'https://iol.invertironline.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbolName': symbol,
            'exchange': 'BCBA',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
            'resolution': 'D',
        }

        response = requests.get(
            'https://iol.invertironline.com/api/cotizaciones/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('status') != 'ok' or 'bars' not in data:
                logger.error(f"Error in API response for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data['bars'])
            df['Date'] = pd.to_datetime(df['time'], unit='s')
            df['Adj Close'] = df['close']
            df = df[['Date', 'Adj Close']]
            df.set_index('Date', inplace=True)
            df = df.sort_index().drop_duplicates()
            return df
        else:
            logger.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error downloading data from IOL for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_byma(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338',
            '_fbp': 'fb.2.1728347943669.954945632708052302',
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,es-AR;q=0.8,es;q=0.7,en-DE;q=0.6,en;q=0.5,en-US;q=0.4',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': 'https://open.bymadata.com.ar/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }

        symbol = ticker.replace('.BA', '') + ' 24HS'

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
            params=params,
            cookies=cookies,
            headers=headers,
            verify=False
        )

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Adj Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            logger.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error downloading data from ByMA Data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def descargar_datos(ticker, start, end, source='YFinance'):
    try:
        if source == 'YFinance':
            return descargar_datos_yfinance(ticker, start, end)
        elif source == 'AnálisisTécnico.com.ar':
            return descargar_datos_analisistecnico(ticker, start, end)
        elif source == 'IOL (Invertir Online)':
            return descargar_datos_iol(ticker, start, end)
        elif source == 'ByMA Data':
            return descargar_datos_byma(ticker, start, end)
        else:
            logger.error(f"Unknown data source: {source}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error downloading data for {ticker} from {source}: {e}")
        return pd.DataFrame()

# ------------------------------
# Helper functions
def ajustar_precios_por_splits(df, ticker):
    try:
        if ticker in splits:
            adjustment = splits[ticker]
            if isinstance(adjustment, tuple):
                split_date = datetime(2023, 11, 3)
                df_before_split = df[df.index < split_date].copy()
                df_after_split = df[df.index >= split_date].copy()
                df_before_split['Adj Close'] /= adjustment[0]
                df_after_split['Adj Close'] *= adjustment[1]
                df = pd.concat([df_before_split, df_after_split]).sort_index()
            else:
                split_threshold_date = datetime(2024, 1, 23)
                df.loc[df.index <= split_threshold_date, 'Adj Close'] /= adjustment
    except Exception as e:
        logger.error(f"Error ajustando splits para {ticker}: {e}")
    return df

@st.cache_data
def load_cpi_data():
    try:
        cpi = pd.read_csv('inflaciónargentina2.csv')
        if 'Date' not in cpi.columns or 'CPI_MoM' not in cpi.columns:
            st.error("El archivo CSV debe contener las columnas 'Date' y 'CPI_MoM'.")
            st.stop()

        cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
        cpi.set_index('Date', inplace=True)
        cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
        daily = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')

        # Ensure index is datetime without timezone
        daily.index = pd.to_datetime(daily.index)
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)

        return daily
    except FileNotFoundError:
        st.error("El archivo 'inflaciónargentina2.csv' no se encontró.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading CPI data: {e}")
        st.stop()

# Load CPI data
daily_cpi = load_cpi_data()

# ------------------------------
# Streamlit UI
st.title('Ajustadora de acciones del Merval por inflación - MTaurus - [X: MTaurus_ok](https://x.com/MTaurus_ok)')

# Add data source selector in sidebar
st.sidebar.title("Configuración")
data_source = st.sidebar.radio(
    "Fuente de datos:",
    ('YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data')
)

# Add information about data sources
st.sidebar.markdown("""
### Información sobre fuentes de datos:
- **YFinance**: Datos internacionales, mejor para tickers extranjeros
- **AnálisisTécnico.com.ar**: Datos locales, mejor para tickers argentinos
- **IOL**: Datos locales con acceso a bonos y otros instrumentos
- **ByMA Data**: Datos oficiales del mercado argentino

*Nota: Algunos tickers pueden no estar disponibles en todas las fuentes.*
""")

# Rest of your UI code...

# ------------------------------
# Calculador de precios por inflación
st.subheader('1- Calculador de precios por inflación')

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

    try:
        start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
        end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
        end_value = start_value * (end_inflation / start_inflation)
        st.write(f"Valor inicial el {start_date}: ARS {start_value}")
        st.write(f"Valor ajustado el {end_date}: ARS {end_value:.2f}")
    except KeyError as e:
        st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
        st.stop()

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

    try:
        start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
        end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
        start_value = end_value / (end_inflation / start_inflation)
        st.write(f"Valor ajustado el {start_date}: ARS {start_value:.2f}")
        st.write(f"Valor final el {end_date}: ARS {end_value}")
    except KeyError as e:
        st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")
        st.stop()

# ------------------------------
# Ajustadora de acciones por inflación
st.subheader('2- Ajustadora de acciones por inflación')

tickers_input = st.text_input(
    'Ingresa los tickers de acciones separados por comas (por ejemplo, AAPL.BA, MSFT.BA, META):',
    key='tickers_input'
)

sma_period = st.number_input(
    'Ingresa el número de periodos para el SMA del primer ticker:',
    min_value=1,
    value=10,
    key='sma_period_input'
)

plot_start_date = st.date_input(
    'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
    min_value=daily_cpi.index.min().date(),
    max_value=daily_cpi.index.max().date(),
    value=(daily_cpi.index.max() - timedelta(days=365)).date(),
    key='plot_start_date_input'
)

show_percentage = st.checkbox('Mostrar valores ajustados por inflación como porcentajes', value=False)
show_percentage_from_recent = st.checkbox(
    'Mostrar valores ajustados por inflación como porcentajes desde el valor más reciente',
    value=False
)

# Diccionarios para almacenar datos
stock_data_dict_nominal = {}
stock_data_dict_adjusted = {}

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    fig = go.Figure()
    ticker_var_map = {ticker: ticker.replace('.', '_') for ticker in tickers}

    # Process each ticker
    # Process each ticker
    # Process each ticker
    # Process each ticker
    # Process each ticker
    # Process each ticker
    for i, ticker in enumerate(tickers):
        try:
            # Download data using selected source
            stock_data = descargar_datos(ticker, plot_start_date, daily_cpi.index.max().date(), data_source)

            if stock_data.empty:
                st.error(f"No se encontraron datos para el ticker {ticker}.")
                continue

            # Ensure datetime index and remove any timezone information
            if 'Date' in stock_data.columns:
                stock_data.set_index('Date', inplace=True)
            stock_data.index = pd.to_datetime(stock_data.index)
            if stock_data.index.tz is not None:
                stock_data.index = stock_data.index.tz_localize(None)

            # For IOL and ByMA, rename the price column to 'Adj Close'
            if data_source in ['IOL (Invertir Online)', 'ByMA Data']:
                # If there's only one column left (excluding Date which is now index)
                if len(stock_data.columns) == 1:
                    stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Adj Close'})

            # Fix timezone offset in index
            stock_data.index = stock_data.index.tz_localize(None)
            stock_data.index = stock_data.index.normalize()  # Remove time component

            # Apply splits adjustment
            stock_data = ajustar_precios_por_splits(stock_data, ticker)

            # Determine if inflation adjustment is needed based on data source and ticker
            needs_inflation_adjustment = (
                    (data_source == 'YFinance' and (ticker.endswith('.BA') or ticker == '^MERV')) or
                    (data_source != 'YFinance')  # Apply inflation adjustment for all tickers from other sources
            )

            if needs_inflation_adjustment and not stock_data.empty:
                # Ensure daily_cpi index is datetime without timezone and normalized
                daily_cpi_clean = daily_cpi.copy()
                daily_cpi_clean.index = pd.to_datetime(daily_cpi_clean.index).normalize()

                # Merge the data
                stock_data = pd.merge(
                    stock_data,
                    daily_cpi_clean,
                    left_index=True,
                    right_index=True,
                    how='left'
                )

                # Forward fill any missing inflation data
                stock_data['Cumulative_Inflation'] = stock_data['Cumulative_Inflation'].ffill().bfill()

                # Calculate inflation adjusted close
                if not stock_data.empty:
                    last_cpi = stock_data['Cumulative_Inflation'].iloc[-1]
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close'] * (
                            last_cpi / stock_data['Cumulative_Inflation']
                    )
                else:
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close']
            else:
                stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close']

            if stock_data.empty:
                st.error(f"No hay datos suficientes para procesar {ticker}.")
                continue

            # Store data
            var_name = ticker_var_map[ticker]
            stock_data_dict_nominal[var_name] = stock_data['Adj Close']
            stock_data_dict_adjusted[var_name] = stock_data['Inflation_Adjusted_Close']

            # Add traces to the figure
            if show_percentage or show_percentage_from_recent:
                if show_percentage_from_recent and len(stock_data) > 0:
                    pct_change = ((stock_data['Inflation_Adjusted_Close'].iloc[-1] /
                                   stock_data['Inflation_Adjusted_Close']) - 1) * 100
                    pct_change = pct_change.clip(lower=-100)
                else:
                    pct_change = (stock_data['Inflation_Adjusted_Close'] /
                                  stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100

                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=pct_change,
                        mode='lines',
                        name=f'{ticker} (%)',
                        yaxis='y1'
                    )
                )

                # Add horizontal line at 0%
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
                # Plot adjusted prices
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Inflation_Adjusted_Close'],
                        mode='lines',
                        name=f'{ticker}',
                        yaxis='y1'
                    )
                )

                # Plot average price
                avg_price = stock_data['Inflation_Adjusted_Close'].mean()
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=[avg_price] * len(stock_data),
                        mode='lines',
                        name=f'{ticker} Precio Promedio',
                        line=dict(dash='dot'),
                        yaxis='y1'
                    )
                )

            # Add SMA for first ticker
            if i == 0 and len(stock_data) > 0:
                stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['SMA'],
                        mode='lines',
                        name=f'{ticker} SMA de {sma_period} Periodos',
                        line=dict(color='orange'),
                        yaxis='y1'
                    )
                )

        except Exception as e:
            st.error(f"Error procesando {ticker}: {e}")
            logger.error(f"Error processing {ticker}: {e}")
            continue

    # Add watermark
    fig.add_annotation(
        text="MTaurus - X: mtaurus_ok",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=30, color="rgba(150, 150, 150, 0.3)"),
        opacity=0.2
    )

    # Update layout
    # Update layout
# Update layout
# Add watermark
fig.add_annotation(
    text="MTaurus - X: mtaurus_ok",
    xref="paper", yref="paper",
    x=0.5, y=0.5,
    showarrow=False,
    font=dict(size=30, color="rgba(150, 150, 150, 0.3)"),
    opacity=0.2
)

# Update layout
if show_percentage or show_percentage_from_recent:
    fig.update_layout(
        title='Precios Históricos Ajustados por Inflación (%)',
        xaxis=dict(title='Fecha'),
        yaxis=dict(title='Variación Porcentual (%)'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
else:
    fig.update_layout(
        title='Precios Históricos Ajustados por Inflación',
        xaxis=dict(title='Fecha'),
        yaxis=dict(title='Precio de Cierre Ajustado (ARS)'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

# Display the plot
st.plotly_chart(fig)

# ------------------------------
# Custom Calculations Section
# ------------------------------
# Custom Calculations Section
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
        # Process custom expression
        sorted_tickers = sorted(ticker_var_map.keys(), key=len, reverse=True)
        transformed_expression = custom_expression
        used_ba_tickers = set()

        # Extract used tickers from the expression
        used_tickers = []
        for ticker in sorted_tickers:
            if ticker in custom_expression:
                used_tickers.append(ticker)
                var_name = ticker_var_map[ticker]
                pattern = re.escape(ticker)
                transformed_expression = re.sub(rf'\b{pattern}\b', var_name, transformed_expression)
                if ticker.endswith('.BA'):
                    used_ba_tickers.add(ticker)

        # Create combined DataFrame only with used tickers
        combined_nominal_df = pd.DataFrame({
            ticker_var_map[ticker]: stock_data_dict_nominal[ticker_var_map[ticker]]
            for ticker in used_tickers
        })

        # Drop rows where any of the used tickers have missing data
        combined_nominal_df.dropna(inplace=True)

        if combined_nominal_df.empty:
            st.error("No hay datos disponibles para todos los tickers seleccionados en las fechas especificadas.")
            st.stop()

        # Evaluate expression
        custom_series_nominal = combined_nominal_df.eval(transformed_expression, engine='python')
        custom_series_nominal = custom_series_nominal.to_frame(name='Custom_Nominal')

        if used_ba_tickers:
            # Apply inflation adjustment only for available dates
            custom_series_nominal = custom_series_nominal.join(daily_cpi, how='inner')
            custom_series_nominal['Cumulative_Inflation'].ffill(inplace=True)
            custom_series_nominal.dropna(subset=['Cumulative_Inflation'], inplace=True)
            custom_series_nominal['Inflation_Adjusted_Custom'] = custom_series_nominal['Custom_Nominal'] * (
                custom_series_nominal['Cumulative_Inflation'].iloc[-1] / custom_series_nominal['Cumulative_Inflation']
            )
            adjusted_series = custom_series_nominal['Inflation_Adjusted_Custom']
        else:
            adjusted_series = custom_series_nominal['Custom_Nominal']

        # Add to plot
        if show_percentage or show_percentage_from_recent:
            if show_percentage_from_recent:
                custom_series_pct = (adjusted_series / adjusted_series.iloc[-1] - 1) * 100
                custom_series_pct = -custom_series_pct
            else:
                custom_series_pct = (adjusted_series / adjusted_series.iloc[0] - 1) * 100

            fig.add_trace(
                go.Scatter(
                    x=custom_series_pct.index,
                    y=custom_series_pct,
                    mode='lines',
                    name=f'Custom: {custom_expression} (%)',
                    yaxis='y1'
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=adjusted_series.index,
                    y=adjusted_series,
                    mode='lines',
                    name=f'Custom: {custom_expression}',
                    yaxis='y1'
                )
            )

        st.plotly_chart(fig)

    except Exception as e:
        available_vars = ', '.join([v for v in ticker_var_map.values()])
        st.error(
            f"Error al evaluar la expresión personalizada: {e}\n\n**Nombres de variables disponibles:** {available_vars}")  

# ------------------------------
# Volatility Analysis Section
st.subheader('4- Comparación de Volatilidad Histórica Ajustada por Inflación y Precio Ajustado por Inflación')

selected_ticker = st.text_input(
    'Ingresa una acción para analizar la volatilidad histórica (puede ser cualquier ticker):',
    placeholder='Por ejemplo: AAPL, AAPL.BA',
    key='selected_ticker_input'
)

vol_comparison_start_date = st.date_input(
    'Selecciona la fecha de inicio para la comparación:',
    min_value=daily_cpi.index.min().date(),
    max_value=daily_cpi.index.max().date(),
    value=(daily_cpi.index.max() - timedelta(days=365)).date(),
    key='vol_comparison_start_date_input'
)

vol_comparison_end_date = st.date_input(
    'Selecciona la fecha de fin para la comparación:',
    min_value=vol_comparison_start_date,
    max_value=daily_cpi.index.max().date(),
    value=daily_cpi.index.max().date(),
    key='vol_comparison_end_date_input'
)

volatility_window = st.number_input(
    'Selecciona el número de periodos para calcular la volatilidad histórica:',
    min_value=1,
    value=20,
    key='volatility_window_input'
)

if selected_ticker:
    ticker = selected_ticker.strip().upper()
    try:
        # Download data using selected source
        stock_data = descargar_datos(ticker, vol_comparison_start_date, vol_comparison_end_date, data_source)

        if not stock_data.empty:
            # Process data
            stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)
            stock_data = ajustar_precios_por_splits(stock_data, ticker)

            if ticker.endswith('.BA'):
                stock_data = stock_data.join(daily_cpi, how='left')
                stock_data['Cumulative_Inflation'].ffill(inplace=True)
                stock_data.dropna(subset=['Cumulative_Inflation'], inplace=True)
                stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close'] * (
                    stock_data['Cumulative_Inflation'].iloc[-1] / stock_data['Cumulative_Inflation']
                )
            else:
                stock_data['Inflation_Adjusted_Close'] = stock_data['Adj Close']

            # Calculate returns and volatility
            stock_data['Return_Adjusted'] = stock_data['Inflation_Adjusted_Close'].pct_change()
            stock_data['Volatility_Adjusted'] = stock_data['Return_Adjusted'].rolling(
                window=volatility_window).std() * (252 ** 0.5)

            # Display volatility
            latest_volatility = stock_data['Volatility_Adjusted'].dropna().iloc[-1]
            st.write(f"**Volatilidad Histórica Ajustada por Inflación (ventana {volatility_window}):** {latest_volatility:.2%}")

            # Create volatility plot
            fig_vol = go.Figure()

            # Add price trace
            fig_vol.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Inflation_Adjusted_Close'],
                    mode='lines',
                    name=f'{ticker} Precio Ajustado por Inflación',
                    line=dict(color='#1f77b4'),
                    yaxis='y1'
                )
            )

            # Add volatility trace
            fig_vol.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Volatility_Adjusted'],
                    mode='lines',
                    name=f'{ticker} Volatilidad Histórica Ajustada',
                    line=dict(color='#ff7f0e'),
                    yaxis='y2'
                )
            )

            # Add watermark
            fig_vol.add_annotation(
                text="MTaurus - X: mtaurus_ok",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=30, color="rgba(150, 150, 150, 0.3)"),
                opacity=0.2
            )

            # Update layout
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

        else:
            st.error(f"No se encontraron datos para el ticker {ticker}.")

    except Exception as e:
        st.error(f"Error al procesar los datos de volatilidad para {ticker}: {e}")
        logger.error(f"Error processing volatility data for {ticker}: {e}")
