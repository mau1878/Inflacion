import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import logging
import re
import requests
import urllib3
import time
import os
from curl_cffi import requests as cffi_requests
from retrying import retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define consistent plot styling for dark mode
plot_style = {
    'plot_bgcolor': 'rgb(30, 30, 30)',
    'paper_bgcolor': 'rgb(30, 30, 30)',
    'font': dict(color='white', family='Arial', size=14),
    'xaxis': dict(
        gridcolor='rgba(255, 255, 255, 0.2)',
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        tickformat="%b %Y",
        tickangle=45,
        nticks=10
    ),
    'yaxis': dict(
        gridcolor='rgba(255, 255, 255, 0.2)',
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        tickformat=".2f",
    ),
    'yaxis2': dict(
        gridcolor='rgba(255, 255, 255, 0.2)',
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        tickformat=".2%"
    ),
    'legend': dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1,
        font=dict(size=12, color='white'),
        bgcolor='rgba(30, 30, 30, 0.8)',
        bordercolor='rgba(255, 255, 255, 0.2)',
        borderwidth=1
    ),
    'template': 'plotly_dark',
    'transition_duration': 0,
    'autosize': True
}

# Color palette for multiple tickers
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

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
@retry(stop_max_attempt_number=3, wait_fixed=5000)
def descargar_datos_yfinance(ticker, start, end):
    try:
        cache_file = f"cache/{ticker}_{start}_{end}.csv"
        os.makedirs("cache", exist_ok=True)

        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=['Date'])
            logger.info(f"Datos cargados desde caché para {ticker}")
            return df

        session = cffi_requests.Session(impersonate="chrome131")
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        stock_data = yf.download(ticker, start=start, end=end, progress=False, session=session)

        if stock_data.empty:
            logger.warning(f"No se encontraron datos para el ticker {ticker} en el rango de fechas seleccionado.")
            return pd.DataFrame()

        if isinstance(stock_data.columns, pd.MultiIndex):
            if 'Close' in stock_data.columns.levels[0]:
                close = stock_data['Close']
                if ticker in close.columns:
                    close = close[ticker].to_frame('Close')
                else:
                    close = close.iloc[:, 0].to_frame('Close')
            else:
                logger.error(f"No 'Close' column found for {ticker} in MultiIndex.")
                return pd.DataFrame()
        else:
            if 'Close' in stock_data.columns:
                close = stock_data[['Close']]
            else:
                logger.error(f"No 'Close' column found for {ticker}.")
                return pd.DataFrame()

        close.to_csv(cache_file)
        logger.info(f"Datos guardados en caché para {ticker}")

        return close

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
                'Close': data['c']
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
            df['Close'] = df['close']
            df = df[['Date', 'Close']]
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
                'Close': data['c']
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
            df = descargar_datos_yfinance(ticker, start, end)
            if df.empty and ticker.endswith('.BA'):
                logger.warning(f"yfinance falló para {ticker}, intentando con analisistecnico...")
                df = descargar_datos_analisistecnico(ticker, start, end)
                if df.empty:
                    logger.warning(f"analisistecnico falló para {ticker}, intentando con iol...")
                    df = descargar_datos_iol(ticker, start, end)
                    if df.empty:
                        logger.warning(f"iol falló para {ticker}, intentando con byma...")
                        df = descargar_datos_byma(ticker, start, end)
        elif source == 'AnálisisTécnico.com.ar':
            df = descargar_datos_analisistecnico(ticker, start, end)
        elif source == 'IOL (Invertir Online)':
            df = descargar_datos_iol(ticker, start, end)
        elif source == 'ByMA Data':
            df = descargar_datos_byma(ticker, start, end)
        else:
            logger.error(f"Unknown data source: {source}")
            return pd.DataFrame()
        time.sleep(5)
        return df
    except Exception as e:
        logger.error(f"Error downloading data for {ticker} from {source}: {e}")
        return pd.DataFrame()

# ------------------------------
# Helper functions
def ajustar_precios_por_splits(df, ticker):
    try:
        if df.empty:
            return df

        df = df.copy()

        all_splits = []

        if ticker in splits:
            adjustment = splits[ticker]
            if isinstance(adjustment, tuple):
                all_splits.append({
                    "date": datetime(2023, 11, 3),
                    "ratio": adjustment[0],
                    "type": "divide"
                })
                all_splits.append({
                    "date": datetime(2023, 11, 3),
                    "ratio": adjustment[1],
                    "type": "multiply"
                })
            else:
                all_splits.append({
                    "date": datetime(2024, 1, 23),
                    "ratio": adjustment,
                    "type": "divide"
                })

        if "custom_splits" in st.session_state:
            for split in st.session_state.custom_splits:
                if split["ticker"] == ticker:
                    all_splits.append({
                        "date": datetime.combine(split["date"], datetime.min.time()),
                        "ratio": split["ratio"],
                        "type": "divide"
                    })

        all_splits = sorted(all_splits, key=lambda x: x["date"])

        for split in all_splits:
            split_date = split["date"]
            ratio = split["ratio"]
            split_type = split["type"]

            if split_type == "divide":
                df.loc[df.index <= split_date, 'Close'] /= ratio
            elif split_type == "multiply":
                df.loc[df.index == split_date, 'Close'] *= ratio

        return df

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

@st.cache_data
def load_us_cpi_data():
    try:
        url = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nUSA.csv"
        cpi = pd.read_csv(url)
        # Assuming similar format to Argentine CPI; adjust if columns differ (e.g., rename if needed)
        if 'Date' not in cpi.columns or 'CPI_MoM' not in cpi.columns:
            st.error("The US CPI CSV must contain columns 'Date' and 'CPI_MoM'. Please check the file.")
            st.stop()

        cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')  # Adjust format if different
        cpi.set_index('Date', inplace=True)
        cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
        daily = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')

        daily.index = pd.to_datetime(daily.index)
        if daily.index.tz is not None:
            daily.index = daily.index.tz_localize(None)

        return daily
    except Exception as e:
        st.error(f"Error loading US CPI data from URL: {e}")
        st.stop()

# Load CPI data
daily_cpi = load_cpi_data()
daily_us_cpi = load_us_cpi_data()

# ------------------------------
# Streamlit UI
st.title('Ajustadora de acciones del Merval por inflación - MTaurus - [X: MTaurus_ok](https://x.com/MTaurus_ok)')

# Sidebar configuration (global)
st.sidebar.title("Configuración")
data_source = st.sidebar.radio(
    "Fuente de datos:",
    ('YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data')
)

st.sidebar.markdown("""
### Información sobre fuentes de datos:
- **YFinance**: Datos internacionales, mejor para tickers extranjeros
- **AnálisisTécnico.com.ar**: Datos locales, mejor para tickers argentinos
- **IOL**: Datos locales con acceso a bonos y otros instrumentos
- **ByMA Data**: Datos oficiales del mercado argentino

*Nota: Algunos tickers pueden no estar disponibles en todas las fuentes.*
""")

st.sidebar.subheader("Ajustes Manuales de Splits")

custom_split_ticker = st.sidebar.text_input(
    "Ingresa el ticker para ajustes de splits (por ejemplo, GLOB.BA):",
    key="custom_split_ticker_input"
)

if "custom_splits" not in st.session_state:
    st.session_state.custom_splits = []

with st.sidebar.form(key="split_form"):
    split_ratio = st.number_input(
        "Ingresa el ratio de split (por ejemplo, 3 para un split 3:1):",
        min_value=1.0,
        step=0.1,
        key="split_ratio_input"
    )
    split_date = st.date_input(
        "Selecciona la fecha del split:",
        min_value=datetime(2000, 1, 1).date(),
        max_value=datetime.now().date(),
        key="split_date_input"
    )
    submit_split = st.form_submit_button("Agregar Split")

    if submit_split and custom_split_ticker:
        st.session_state.custom_splits.append({
            "ticker": custom_split_ticker.strip().upper(),
            "ratio": split_ratio,
            "date": split_date
        })
        st.sidebar.success(f"Split agregado: {split_ratio} en {split_date} para {custom_split_ticker}")

if st.session_state.custom_splits:
    st.sidebar.write("Splits Personalizados Agregados:")
    for i, split in enumerate(st.session_state.custom_splits):
        st.sidebar.write(
            f"Ticker: {split['ticker']}, Ratio: {split['ratio']}, Fecha: {split['date']}"
        )
        if st.sidebar.button(f"Eliminar Split {i+1}", key=f"remove_split_{i}"):
            st.session_state.custom_splits.pop(i)
            st.sidebar.success("Split eliminado.")

st.sidebar.subheader("Eventos Personalizados")

if "custom_events" not in st.session_state:
    st.session_state.custom_events = []

with st.sidebar.form(key="event_form"):
    event_ticker = st.text_input(
        "Ingresa el ticker para el evento (por ejemplo, GLOB.BA):",
        key="event_ticker_input"
    )
    event_date = st.date_input(
        "Selecciona la fecha del evento:",
        min_value=datetime(2000, 1, 1).date(),
        max_value=datetime.now().date(),
        key="event_date_input"
    )
    event_description = st.text_input(
        "Ingresa una descripción para el evento (por ejemplo, Ganancias Q4):",
        key="event_description_input"
    )
    submit_event = st.form_submit_button("Agregar Evento")

    if submit_event and event_ticker and event_description:
        st.session_state.custom_events.append({
            "ticker": event_ticker.strip().upper(),
            "date": event_date,
            "description": event_description
        })
        st.sidebar.success(f"Evento agregado: {event_description} en {event_date} para {event_ticker}")

if st.session_state.custom_events:
    st.sidebar.write("Eventos Personalizados Agregados:")
    for i, event in enumerate(st.session_state.custom_events):
        st.sidebar.write(
            f"Ticker: {event['ticker']}, Evento: {event['description']}, Fecha: {event['date']}"
        )
        if st.sidebar.button(f"Eliminar Evento {i+1}", key=f"remove_event_{i}"):
            st.session_state.custom_events.pop(i)
            st.sidebar.success("Evento eliminado.")

# Main content in tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Inflation Calculator", "Argentine Stock Adjuster", "Custom Calculations", "Volatility Analysis", "US Stock Adjuster"])

with tab1:
    st.subheader('Calculador de precios por inflación (Argentina)')

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

with tab2:
    st.subheader('Ajustadora de acciones por inflación (Argentina)')

    tickers_input = st.text_input(
        'Ingresa los tickers de acciones separados por comas (por ejemplo, AAPL.BA, MSFT.BA, META):',
        key='tickers_input_arg'
    )

    sma_period = st.number_input(
        'Ingresa el número de periodos para el SMA del primer ticker:',
        min_value=1,
        value=10,
        key='sma_period_input_arg'
    )

    plot_start_date = st.date_input(
        'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
        min_value=daily_cpi.index.min().date(),
        max_value=daily_cpi.index.max().date(),
        value=(daily_cpi.index.max() - timedelta(days=365)).date(),
        key='plot_start_date_input_arg'
    )

    show_percentage = st.checkbox('Mostrar valores ajustados por inflación como porcentajes', value=False, key='show_percentage_arg')
    show_percentage_from_recent = st.checkbox(
        'Mostrar valores ajustados por inflación como porcentajes desde el valor más reciente',
        value=False,
        key='show_percentage_from_recent_arg'
    )

    is_percentage_mode = show_percentage or show_percentage_from_recent
    if not is_percentage_mode:
        use_log_scale_arg = st.checkbox('Usar escala logarítmica en el eje Y', value=False, key='use_log_scale_arg')
    else:
        use_log_scale_arg = False

    # Diccionarios para almacenar datos (for Argentine tab)
    stock_data_dict_nominal_arg = {}
    stock_data_dict_adjusted_arg = {}

    if tickers_input:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        fig = go.Figure()
        ticker_var_map = {ticker: ticker.replace('.', '_') for ticker in tickers}

        for i, ticker in enumerate(tickers):
            try:
                stock_data = descargar_datos(ticker, plot_start_date, daily_cpi.index.max().date() + timedelta(days=1), data_source)

                if stock_data.empty:
                    st.error(f"No se encontraron datos para el ticker {ticker}.")
                    continue

                if 'Date' in stock_data.columns:
                    stock_data.set_index('Date', inplace=True)
                stock_data.index = pd.to_datetime(stock_data.index)
                if stock_data.index.tz is not None:
                    stock_data.index = stock_data.index.tz_localize(None)

                if data_source in ['IOL (Invertir Online)', 'ByMA Data']:
                    if len(stock_data.columns) == 1:
                        stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Close'})

                stock_data.index = stock_data.index.tz_localize(None)
                stock_data.index = stock_data.index.normalize()

                stock_data = ajustar_precios_por_splits(stock_data, ticker)

                needs_inflation_adjustment = (
                    (data_source == 'YFinance' and (ticker.endswith('.BA') or ticker == '^MERV')) or
                    (data_source != 'YFinance')
                )

                if needs_inflation_adjustment and not stock_data.empty:
                    daily_cpi_clean = daily_cpi.copy()
                    daily_cpi_clean.index = pd.to_datetime(daily_cpi_clean.index).normalize()
                    stock_data = pd.merge(
                        stock_data,
                        daily_cpi_clean,
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                    stock_data['Cumulative_Inflation'] = stock_data['Cumulative_Inflation'].ffill().bfill()
                    if not stock_data.empty:
                        last_cpi = stock_data['Cumulative_Inflation'].iloc[-1]
                        stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (
                            last_cpi / stock_data['Cumulative_Inflation']
                        )
                    else:
                        stock_data['Inflation_Adjusted_Close'] = stock_data['Close']
                else:
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Close']

                if stock_data.empty:
                    st.error(f"No hay datos suficientes para procesar {ticker}.")
                    continue

                var_name = ticker_var_map[ticker]
                stock_data_dict_nominal_arg[var_name] = stock_data['Close']
                stock_data_dict_adjusted_arg[var_name] = stock_data['Inflation_Adjusted_Close']

                display_name = f'{ticker[:10]}...' if len(ticker) > 10 else ticker

                if is_percentage_mode:
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
                            name=f'{display_name} (%)',
                            line=dict(color=colors[i % len(colors)], width=1.5),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación: %{y:.2f}%<extra></extra>'
                        )
                    )
                    fig.add_shape(
                        type="line",
                        x0=stock_data.index.min(),
                        x1=stock_data.index.max(),
                        y0=0,
                        y1=0,
                        line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
                        xref="x",
                        yref="y1"
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Inflation_Adjusted_Close'],
                            mode='lines',
                            name=display_name,
                            line=dict(color=colors[i % len(colors)], width=1.5),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f} ARS<extra></extra>'
                        )
                    )
                    avg_price = stock_data['Inflation_Adjusted_Close'].mean()
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=[avg_price] * len(stock_data),
                            mode='lines',
                            name=f'{display_name} Avg',
                            line=dict(color=colors[i % len(colors)], width=0.8, dash='dot'),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Promedio: %{y:.2f} ARS<extra></extra>'
                        )
                    )

                if i == 0 and len(stock_data) > 0:
                    stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=stock_data['SMA'],
                            mode='lines',
                            name=f'{display_name} SMA',
                            line=dict(color='orange', width=1),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>SMA: %{y:.2f} ARS<extra></extra>'
                        )
                    )

                # Add split annotations
                for split in st.session_state.custom_splits:
                    if split["ticker"] == ticker:
                        fig.add_vline(
                            x=datetime.combine(split["date"], datetime.min.time()).timestamp() * 1000,
                            line=dict(color="white", width=1, dash="dash"),
                            annotation_text=f"Split {split['ratio']}:1",
                            annotation_position="top",
                            annotation=dict(font=dict(color='white'))
                        )
                # Add event annotations
                for event in st.session_state.custom_events:
                    if event["ticker"] == ticker:
                        fig.add_vline(
                            x=datetime.combine(event["date"], datetime.min.time()).timestamp() * 1000,
                            line=dict(color="yellow", width=1, dash="dot"),
                            annotation_text=event["description"],
                            annotation_position="top",
                            annotation=dict(font=dict(color='yellow'))
                        )
            except Exception as e:
                st.error(f"Error procesando {ticker}: {e}")
                logger.error(f"Error processing {ticker}: {e}")
                continue

        fig.add_annotation(
            text="MTaurus - X: mtaurus_ok",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=30, color="rgba(255, 255, 255, 0.2)"),
            opacity=0.15
        )

        fig.update_layout(
            title=dict(
                text='Precios Históricos Ajustados por Inflación' if not is_percentage_mode else 'Precios Históricos Ajustados por Inflación (%)',
                font=dict(size=20, color='white')
            ),
            xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
            yaxis_title=dict(
                text='Precio de Cierre Ajustado (ARS)' if not is_percentage_mode else 'Variación Porcentual (%)',
                font=dict(size=14, color='white')
            ),
            **plot_style
        )

        # Update yaxis for log scale and formatting
        fig.update_yaxes(
            type='log' if use_log_scale_arg else 'linear',
            tickformat=',.2f' if not is_percentage_mode else ',.2f',
            ticksuffix='' if not is_percentage_mode else '%'
        )

        st.plotly_chart(fig)

with tab3:
    st.subheader('Cálculos o Ratios Personalizados')

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
            sorted_tickers = sorted(ticker_var_map.keys(), key=len, reverse=True)
            transformed_expression = custom_expression
            used_ba_tickers = set()

            used_tickers = []
            for ticker in sorted_tickers:
                if ticker in custom_expression:
                    used_tickers.append(ticker)
                    var_name = ticker_var_map[ticker]
                    pattern = re.escape(ticker)
                    transformed_expression = re.sub(rf'\b{pattern}\b', var_name, transformed_expression)
                    if ticker.endswith('.BA'):
                        used_ba_tickers.add(ticker)

            combined_nominal_df = pd.DataFrame({
                ticker_var_map[ticker]: stock_data_dict_nominal_arg[ticker_var_map[ticker]]
                for ticker in used_tickers
            })

            combined_nominal_df.dropna(inplace=True)

            if combined_nominal_df.empty:
                st.error("No hay datos disponibles para todos los tickers seleccionados en las fechas especificadas.")
            else:
                custom_series_nominal = combined_nominal_df.eval(transformed_expression, engine='python')
                custom_series_nominal = custom_series_nominal.to_frame(name='Custom_Nominal')

                if used_ba_tickers:
                    custom_series_nominal = custom_series_nominal.join(daily_cpi, how='inner')
                    custom_series_nominal['Cumulative_Inflation'].ffill(inplace=True)
                    custom_series_nominal.dropna(subset=['Cumulative_Inflation'], inplace=True)
                    custom_series_nominal['Inflation_Adjusted_Custom'] = custom_series_nominal['Custom_Nominal'] * (
                        custom_series_nominal['Cumulative_Inflation'].iloc[-1] / custom_series_nominal['Cumulative_Inflation']
                    )
                    adjusted_series = custom_series_nominal['Inflation_Adjusted_Custom']
                else:
                    adjusted_series = custom_series_nominal['Custom_Nominal']

                fig = go.Figure()  # New figure for custom

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
                            name=f'Custom: {custom_expression[:10]}...' if len(custom_expression) > 10 else f'Custom: {custom_expression}',
                            line=dict(color=colors[-1], width=1.5),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación: %{y:.2f}%<extra></extra>'
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=adjusted_series.index,
                            y=adjusted_series,
                            mode='lines',
                            name=f'Custom: {custom_expression[:10]}...' if len(custom_expression) > 10 else f'Custom: {custom_expression}',
                            line=dict(color=colors[-1], width=1.5),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Valor: %{y:.2f} ARS<extra></extra>'
                        )
                    )

                fig.update_layout(
                    title=dict(
                        text='Precios Históricos Ajustados por Inflación' if not (show_percentage or show_percentage_from_recent) else 'Precios Históricos Ajustados por Inflación (%)',
                        font=dict(size=20, color='white')
                    ),
                    xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
                    yaxis_title=dict(
                        text='Precio de Cierre Ajustado (ARS)' if not (show_percentage or show_percentage_from_recent) else 'Variación Porcentual (%)',
                        font=dict(size=14, color='white')
                    ),
                    **plot_style
                )

                st.plotly_chart(fig)

        except Exception as e:
            available_vars = ', '.join([v for v in ticker_var_map.values()])
            st.error(
                f"Error al evaluar la expresión personalizada: {e}\n\n**Nombres de variables disponibles:** {available_vars}")

with tab4:
    st.subheader('Comparación de Volatilidad Histórica Ajustada por Inflación y Precio Ajustado por Inflación (Argentina)')

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
            stock_data = descargar_datos(ticker, vol_comparison_start_date, vol_comparison_end_date, data_source)

            if not stock_data.empty:
                stock_data.index = pd.to_datetime(stock_data.index).tz_localize(None)
                stock_data = ajustar_precios_por_splits(stock_data, ticker)

                if ticker.endswith('.BA'):
                    stock_data = stock_data.join(daily_cpi, how='left')
                    stock_data['Cumulative_Inflation'].ffill(inplace=True)
                    stock_data.dropna(subset=['Cumulative_Inflation'], inplace=True)
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (
                        stock_data['Cumulative_Inflation'].iloc[-1] / stock_data['Cumulative_Inflation']
                    )
                else:
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Close']

                stock_data['Return_Adjusted'] = stock_data['Inflation_Adjusted_Close'].pct_change()
                stock_data['Volatility_Adjusted'] = stock_data['Return_Adjusted'].rolling(
                    window=volatility_window).std() * (252 ** 0.5)

                latest_volatility = stock_data['Volatility_Adjusted'].dropna().iloc[-1]
                st.write(f"**Volatilidad Histórica Ajustada por Inflación (ventana {volatility_window}):** {latest_volatility:.2%}")

                fig_vol = go.Figure()

                display_name = f'{ticker[:10]}...' if len(ticker) > 10 else ticker

                fig_vol.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Inflation_Adjusted_Close'],
                        mode='lines',
                        name=f'{display_name} Precio Ajustado',
                        line=dict(color=colors[0], width=1.5),
                        yaxis='y1',
                        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f} ARS<extra></extra>'
                    )
                )

                fig_vol.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Volatility_Adjusted'],
                        mode='lines',
                        name=f'{display_name} Volatilidad',
                        line=dict(color=colors[1], width=1.5),
                        yaxis='y2',
                        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Volatilidad: %{y:.2%}<extra></extra>'
                    )
                )

                for split in st.session_state.custom_splits:
                    if split["ticker"] == ticker:
                        fig_vol.add_vline(
                            x=datetime.combine(split["date"], datetime.min.time()).timestamp() * 1000,
                            line=dict(color="white", width=1, dash="dash"),
                            annotation_text=f"Split {split['ratio']}:1",
                            annotation_position="top",
                            annotation=dict(font=dict(color='white'))
                        )

                fig_vol.add_annotation(
                    text="MTaurus - X: mtaurus_ok",
                    xref="paper", yref="paper",
                    x=0.02, y=0.02,
                    showarrow=False,
                    font=dict(size=20, color="rgba(255, 255, 255, 0.2)"),
                    opacity=0.1
                )

                fig_vol.update_layout(
                    title=dict(
                        text=f'Precio Ajustado por Inflación y Volatilidad Histórica de {display_name}',
                        font=dict(size=20, color='white')
                    ),
                    xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
                    yaxis=dict(
                        title='Precio de Cierre Ajustado (ARS)',
                        titlefont=dict(color='white', size=14),
                        tickfont=dict(color='white'),
                        tickformat=",.2f",
                        ticksuffix=" ARS"
                    ),
                    yaxis2=dict(
                        title='Volatilidad Histórica (Anualizada)',
                        titlefont=dict(color='white', size=14),
                        tickfont=dict(color='white'),
                        tickformat=".2%",
                        overlaying='y',
                        side='right'
                    ),
                    **plot_style
                )

                st.plotly_chart(fig_vol)

            else:
                st.error(f"No se encontraron datos para el ticker {ticker}.")

        except Exception as e:
            st.error(f"Error al procesar los datos de volatilidad para {ticker}: {e}")
            logger.error(f"Error processing volatility data for {ticker}: {e}")

with tab5:
    st.subheader('Ajustadora de acciones por inflación (USA)')

    tickers_input_us = st.text_input(
        'Ingresa los tickers de acciones separados por comas (por ejemplo, AAPL, MSFT, META.BA):',
        key='tickers_input_us'
    )

    sma_period_us = st.number_input(
        'Ingresa el número de periodos para el SMA del primer ticker:',
        min_value=1,
        value=10,
        key='sma_period_input_us'
    )

    plot_start_date_us = st.date_input(
        'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
        min_value=daily_us_cpi.index.min().date(),
        max_value=daily_us_cpi.index.max().date(),
        value=(daily_us_cpi.index.max() - timedelta(days=365)).date(),
        key='plot_start_date_input_us'
    )

    show_percentage_us = st.checkbox('Mostrar valores ajustados por inflación como porcentajes', value=False, key='show_percentage_us')
    show_percentage_from_recent_us = st.checkbox(
        'Mostrar valores ajustados por inflación como porcentajes desde el valor más reciente',
        value=False,
        key='show_percentage_from_recent_us'
    )

    is_percentage_mode_us = show_percentage_us or show_percentage_from_recent_us
    if not is_percentage_mode_us:
        use_log_scale_us = st.checkbox('Usar escala logarítmica en el eje Y', value=False, key='use_log_scale_us')
    else:
        use_log_scale_us = False

    # Diccionarios para almacenar datos (for US tab)
    stock_data_dict_nominal_us = {}
    stock_data_dict_adjusted_us = {}

    if tickers_input_us:
        tickers = [ticker.strip().upper() for ticker in tickers_input_us.split(',')]
        fig_us = go.Figure()
        ticker_var_map_us = {ticker: ticker.replace('.', '_') for ticker in tickers}

        for i, ticker in enumerate(tickers):
            try:
                stock_data = descargar_datos(ticker, plot_start_date_us, daily_us_cpi.index.max().date() + timedelta(days=1), data_source)

                if stock_data.empty:
                    st.error(f"No se encontraron datos para el ticker {ticker}.")
                    continue

                if 'Date' in stock_data.columns:
                    stock_data.set_index('Date', inplace=True)
                stock_data.index = pd.to_datetime(stock_data.index)
                if stock_data.index.tz is not None:
                    stock_data.index = stock_data.index.tz_localize(None)

                if data_source in ['IOL (Invertir Online)', 'ByMA Data']:
                    if len(stock_data.columns) == 1:
                        stock_data = stock_data.rename(columns={stock_data.columns[0]: 'Close'})

                stock_data.index = stock_data.index.tz_localize(None)
                stock_data.index = stock_data.index.normalize()

                stock_data = ajustar_precios_por_splits(stock_data, ticker)

                # For US tab, always apply inflation adjustment (change if needed)
                if not stock_data.empty:
                    daily_us_cpi_clean = daily_us_cpi.copy()
                    daily_us_cpi_clean.index = pd.to_datetime(daily_us_cpi_clean.index).normalize()
                    stock_data = pd.merge(
                        stock_data,
                        daily_us_cpi_clean,
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                    stock_data['Cumulative_Inflation'] = stock_data['Cumulative_Inflation'].ffill().bfill()
                    if not stock_data.empty:
                        last_cpi = stock_data['Cumulative_Inflation'].iloc[-1]
                        stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (
                            last_cpi / stock_data['Cumulative_Inflation']
                        )
                    else:
                        stock_data['Inflation_Adjusted_Close'] = stock_data['Close']
                else:
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Close']

                if stock_data.empty:
                    st.error(f"No hay datos suficientes para procesar {ticker}.")
                    continue

                var_name = ticker_var_map_us[ticker]
                stock_data_dict_nominal_us[var_name] = stock_data['Close']
                stock_data_dict_adjusted_us[var_name] = stock_data['Inflation_Adjusted_Close']

                display_name = f'{ticker[:10]}...' if len(ticker) > 10 else ticker

                if is_percentage_mode_us:
                    if show_percentage_from_recent_us and len(stock_data) > 0:
                        pct_change = ((stock_data['Inflation_Adjusted_Close'].iloc[-1] /
                                       stock_data['Inflation_Adjusted_Close']) - 1) * 100
                        pct_change = pct_change.clip(lower=-100)
                    else:
                        pct_change = (stock_data['Inflation_Adjusted_Close'] /
                                      stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100

                    fig_us.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=pct_change,
                            mode='lines',
                            name=f'{display_name} (%)',
                            line=dict(color=colors[i % len(colors)], width=1.5),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación: %{y:.2f}%<extra></extra>'
                        )
                    )
                    fig_us.add_shape(
                        type="line",
                        x0=stock_data.index.min(),
                        x1=stock_data.index.max(),
                        y0=0,
                        y1=0,
                        line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
                        xref="x",
                        yref="y1"
                    )
                else:
                    fig_us.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Inflation_Adjusted_Close'],
                            mode='lines',
                            name=display_name,
                            line=dict(color=colors[i % len(colors)], width=1.5),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f} USD<extra></extra>'  # Changed to USD for US context
                        )
                    )
                    avg_price = stock_data['Inflation_Adjusted_Close'].mean()
                    fig_us.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=[avg_price] * len(stock_data),
                            mode='lines',
                            name=f'{display_name} Avg',
                            line=dict(color=colors[i % len(colors)], width=0.8, dash='dot'),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Promedio: %{y:.2f} USD<extra></extra>'
                        )
                    )

                if i == 0 and len(stock_data) > 0:
                    stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period_us).mean()
                    fig_us.add_trace(
                        go.Scatter(
                            x=stock_data.index,
                            y=stock_data['SMA'],
                            mode='lines',
                            name=f'{display_name} SMA',
                            line=dict(color='orange', width=1),
                            yaxis='y1',
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>SMA: %{y:.2f} USD<extra></extra>'
                        )
                    )

                # Add split annotations
                for split in st.session_state.custom_splits:
                    if split["ticker"] == ticker:
                        fig_us.add_vline(
                            x=datetime.combine(split["date"], datetime.min.time()).timestamp() * 1000,
                            line=dict(color="white", width=1, dash="dash"),
                            annotation_text=f"Split {split['ratio']}:1",
                            annotation_position="top",
                            annotation=dict(font=dict(color='white'))
                        )
                # Add event annotations
                for event in st.session_state.custom_events:
                    if event["ticker"] == ticker:
                        fig_us.add_vline(
                            x=datetime.combine(event["date"], datetime.min.time()).timestamp() * 1000,
                            line=dict(color="yellow", width=1, dash="dot"),
                            annotation_text=event["description"],
                            annotation_position="top",
                            annotation=dict(font=dict(color='yellow'))
                        )
            except Exception as e:
                st.error(f"Error procesando {ticker}: {e}")
                logger.error(f"Error processing {ticker}: {e}")
                continue

        fig_us.add_annotation(
            text="MTaurus - X: mtaurus_ok",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=30, color="rgba(255, 255, 255, 0.2)"),
            opacity=0.15
        )

        fig_us.update_layout(
            title=dict(
                text='Precios Históricos Ajustados por Inflación (USA)' if not is_percentage_mode_us else 'Precios Históricos Ajustados por Inflación (USA) (%)',
                font=dict(size=20, color='white')
            ),
            xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
            yaxis_title=dict(
                text='Precio de Cierre Ajustado (USD)' if not is_percentage_mode_us else 'Variación Porcentual (%)',
                font=dict(size=14, color='white')
            ),
            **plot_style
        )

        # Update yaxis for log scale and formatting
        fig_us.update_yaxes(
            type='log' if use_log_scale_us else 'linear',
            tickformat=',.2f' if not is_percentage_mode_us else ',.2f',
            ticksuffix='' if not is_percentage_mode_us else '%'
        )

        st.plotly_chart(fig_us)
