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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from num2words import num2words

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

# Estilo oscuro para los gráficos Matplotlib/Seaborn (más prácticos para manejar desde el celular)
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#1e1e1e',
    'axes.facecolor': '#1e1e1e',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': 'white',
    'grid.alpha': 0.2,
    'legend.facecolor': '#1e1e1e',
    'legend.edgecolor': 'white',
    'legend.labelcolor': 'white',
    'font.size': 10,
})


def _finalizar_grafico_mpl(fig, ax, titulo, ylabel_txt, is_percentage, use_log_scale, ax2=None):
    """Aplica estilo consistente (oscuro, marca de agua, formato de fechas) a un gráfico Matplotlib/Seaborn.
    ax2, si se pasa, es un eje secundario (twinx) cuyas líneas se suman a la leyenda combinada."""
    ax.set_title(titulo, fontsize=15, color='white', pad=12)
    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel(ylabel_txt, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=45)
    if is_percentage:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.2f}'))
        if use_log_scale:
            ax.set_yscale('log')
    ax.text(
        0.5, 0.5, "MTaurus - X: mtaurus_ok", transform=ax.transAxes,
        fontsize=24, color='white', alpha=0.12, ha='center', va='center'
    )
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles, labels = handles + handles2, labels + labels2
    if handles:
        ncols = min(len(labels), 3)
        filas_leyenda = -(-len(labels) // ncols)  # redondeo hacia arriba
        espacio_inferior = 0.22 + 0.06 * filas_leyenda
        fig.subplots_adjust(bottom=espacio_inferior)
        ax.legend(
            handles, labels,
            loc='upper center', bbox_to_anchor=(0.5, -espacio_inferior * 1.35),
            ncol=ncols, fontsize=8
        )


def add_marker_lines(fig, items, color, y_top, name, dash="dot"):
    """
    Dibuja líneas verticales SIN texto fijo (evita el amontonamiento cuando hay
    muchos eventos/splits cercanos) y agrega un marcador con la info solo al
    pasar el mouse (hover) sobre el punto, ubicado cerca del techo de la serie.
    `items` es una lista de tuplas (fecha_datetime, texto_descripcion).
    """
    if not items:
        return
    for dt, _ in items:
        fig.add_vline(x=dt.timestamp() * 1000, line=dict(color=color, width=1, dash=dash))
    fig.add_trace(
        go.Scatter(
            x=[dt for dt, _ in items],
            y=[y_top] * len(items),
            mode='markers',
            marker=dict(symbol='triangle-down', size=9, color=color, line=dict(width=1, color='black')),
            hovertext=[f"{texto}<br>{dt.strftime('%Y-%m-%d')}" for dt, texto in items],
            hoverinfo='text',
            name=name,
            showlegend=False,
        )
    )


def graficar_activos_ajustados(
    tickers_input,
    sma_period,
    plot_start_date,
    daily_cpi_serie,
    data_source,
    moneda,
    is_percentage_mode,
    show_percentage_from_recent,
    use_log_scale,
    show_nominal_ghost,
    siempre_ajustar=False,
    force_inflation=False,
    plot_end_date=None,
    sma_line_width=1.0,
    data_line_width=1.5,
    show_mep_ghost=False,
):
    """
    Descarga, ajusta por inflación y grafica (Plotly + Matplotlib/Seaborn) una lista de tickers.
    moneda: 'ARS' o 'USD', usado solo para etiquetas/hovers.
    siempre_ajustar=True implica que se ajusta siempre por inflación (pestaña EEUU).
    force_inflation solo aplica cuando siempre_ajustar=False (pestaña Argentina).
    show_nominal_ghost agrega, por ticker, una línea punteada "fantasma" con el valor nominal
    (sin ajustar por inflación), tanto en modo absoluto como en modo porcentual.
    show_mep_ghost agrega, por ticker (solo en modo absoluto), una línea "fantasma" con el
    precio nominal convertido a USD MEP, en un eje Y secundario.
    plot_end_date permite fijar la fecha final del rango (por defecto, la fecha actual /
    el último dato de IPC disponible, igual que antes).
    Devuelve (stock_data_dict_nominal, stock_data_dict_adjusted, ticker_var_map).
    """
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
    fig = go.Figure()
    fig_mpl, ax_mpl = plt.subplots(figsize=(11, 5.5))
    ax_mpl2 = None

    ticker_var_map = {ticker: ticker.replace('.', '_') for ticker in tickers}
    stock_data_dict_nominal = {}
    stock_data_dict_adjusted = {}

    if plot_end_date is None:
        end_date = daily_cpi_serie.index.max().date() + timedelta(days=1)
    else:
        end_date = plot_end_date + timedelta(days=1)

    for i, ticker in enumerate(tickers):
        try:
            stock_data = descargar_datos(ticker, plot_start_date, end_date, data_source)

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
            stock_data = ajustar_precios_por_cupones(stock_data, ticker, cashflows_bonos, daily_mep)

            if siempre_ajustar:
                needs_inflation_adjustment = True
            else:
                needs_inflation_adjustment = force_inflation or (
                    (data_source == 'YFinance' and (ticker.endswith('.BA') or ticker == '^MERV')) or
                    (data_source != 'YFinance')
                )

            if needs_inflation_adjustment and not stock_data.empty:
                daily_cpi_clean = daily_cpi_serie.copy()
                daily_cpi_clean.index = pd.to_datetime(daily_cpi_clean.index).normalize()
                stock_data = pd.merge(
                    stock_data, daily_cpi_clean,
                    left_index=True, right_index=True, how='left'
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

            # --- DEBUG temporal: comparar Close vs Inflation_Adjusted_Close
            # justo antes/después de cada fecha ex-cupón detectada ---
            ticker_csv_debug = _resolver_ticker_bono(ticker, cashflows_bonos) if not cashflows_bonos.empty else None
            if ticker_csv_debug is not None:
                pagos_debug = cashflows_bonos[cashflows_bonos['Bono'] == ticker_csv_debug].sort_values('Fecha')
                filas_debug = []
                for _, pago_d in pagos_debug.iterrows():
                    fecha_pago_d = pd.Timestamp(pago_d['Fecha'])
                    if fecha_pago_d > stock_data.index.max():
                        continue
                    dias_prev_d = stock_data.index[stock_data.index < fecha_pago_d]
                    if dias_prev_d.empty:
                        continue
                    fecha_ex_d = dias_prev_d.max()
                    dias_prev_ex_d = stock_data.index[stock_data.index < fecha_ex_d]
                    if dias_prev_ex_d.empty:
                        continue
                    fecha_prev_d = dias_prev_ex_d.max()
                    filas_debug.append({
                        'fecha_prev': fecha_prev_d.date(),
                        'Close_prev': stock_data.loc[fecha_prev_d, 'Close'],
                        'InflAdj_prev': stock_data.loc[fecha_prev_d, 'Inflation_Adjusted_Close'],
                        'fecha_ex': fecha_ex_d.date(),
                        'Close_ex': stock_data.loc[fecha_ex_d, 'Close'],
                        'InflAdj_ex': stock_data.loc[fecha_ex_d, 'Inflation_Adjusted_Close'],
                    })
                if filas_debug:
                    st.caption(f"🔎 DEBUG {ticker}: comparación Close / Inflation_Adjusted_Close alrededor de cada ex-cupón")
                    st.dataframe(pd.DataFrame(filas_debug))
            # --- FIN DEBUG temporal ---

            if stock_data.empty:
                st.error(f"No hay datos suficientes para procesar {ticker}.")
                continue

            var_name = ticker_var_map[ticker]
            stock_data_dict_nominal[var_name] = stock_data['Close']
            stock_data_dict_adjusted[var_name] = stock_data['Inflation_Adjusted_Close']

            display_name = f'{ticker[:10]}...' if len(ticker) > 10 else ticker
            color = colors[i % len(colors)]

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
                        x=stock_data.index, y=pct_change, mode='lines',
                        name=f'{display_name} (Ajustado, %)',
                        line=dict(color=color, width=data_line_width), yaxis='y1',
                        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación: %{y:.2f}%<extra></extra>'
                    )
                )
                fig.add_shape(
                    type="line", x0=stock_data.index.min(), x1=stock_data.index.max(),
                    y0=0, y1=0, line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
                    xref="x", yref="y1"
                )
                ax_mpl.plot(stock_data.index, pct_change, color=color, linewidth=data_line_width, label=f'{display_name} (Ajustado, %)')
                ax_mpl.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)

                if show_nominal_ghost:
                    if show_percentage_from_recent and len(stock_data) > 0:
                        pct_change_nom = ((stock_data['Close'].iloc[-1] / stock_data['Close']) - 1) * 100
                        pct_change_nom = pct_change_nom.clip(lower=-100)
                    else:
                        pct_change_nom = (stock_data['Close'] / stock_data['Close'].iloc[0] - 1) * 100

                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index, y=pct_change_nom, mode='lines',
                            name=f'{display_name} Nominal (%)',
                            line=dict(color=color, width=1, dash='dot'), yaxis='y1', opacity=0.55,
                            hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación nominal: %{y:.2f}%<extra></extra>'
                        )
                    )
                    ax_mpl.plot(
                        stock_data.index, pct_change_nom, color=color, linewidth=1,
                        linestyle=':', alpha=0.55, label=f'{display_name} Nominal (%)'
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'], mode='lines',
                        name=f'{display_name} (Ajustado por Inflación)', line=dict(color=color, width=data_line_width), yaxis='y1',
                        hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>Precio: %{{y:.2f}} {moneda}<extra></extra>'
                    )
                )
                avg_price = stock_data['Inflation_Adjusted_Close'].mean()
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index, y=[avg_price] * len(stock_data), mode='lines',
                        name=f'{display_name} Promedio (Ajustado)', line=dict(color=color, width=0.8, dash='dot'), yaxis='y1',
                        hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>Promedio: %{{y:.2f}} {moneda}<extra></extra>'
                    )
                )
                ax_mpl.plot(
                    stock_data.index, stock_data['Inflation_Adjusted_Close'],
                    color=color, linewidth=data_line_width, label=f'{display_name} (Ajustado por Inflación)'
                )
                ax_mpl.axhline(avg_price, color=color, linewidth=0.8, linestyle=':', alpha=0.8)

                if show_nominal_ghost:
                    fig.add_trace(
                        go.Scatter(
                            x=stock_data.index, y=stock_data['Close'], mode='lines',
                            name=f'{display_name} Nominal', line=dict(color=color, width=1, dash='dot'),
                            yaxis='y1', opacity=0.5,
                            hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>Nominal: %{{y:.2f}} {moneda}<extra></extra>'
                        )
                    )
                    ax_mpl.plot(
                        stock_data.index, stock_data['Close'], color=color, linewidth=1,
                        linestyle=':', alpha=0.5, label=f'{display_name} Nominal'
                    )

                if show_mep_ghost and moneda == 'ARS' and daily_mep is not None and not daily_mep.empty:
                    mep_alineado = daily_mep.reindex(stock_data.index).ffill()
                    if mep_alineado.notna().any():
                        stock_data['Close_MEP'] = stock_data['Close'] / mep_alineado
                        fig.add_trace(
                            go.Scatter(
                                x=stock_data.index, y=stock_data['Close_MEP'], mode='lines',
                                name=f'{display_name} (USD MEP)',
                                line=dict(color=color, width=1, dash='dashdot'),
                                yaxis='y2', opacity=0.6,
                                hovertemplate='Fecha: %{x|%Y-%m-%d}<br>USD MEP: %{y:.2f}<extra></extra>'
                            )
                        )
                        if ax_mpl2 is None:
                            ax_mpl2 = ax_mpl.twinx()
                            ax_mpl2.set_ylabel('Precio en USD (MEP)', color='white')
                            ax_mpl2.tick_params(axis='y', colors='white')
                        ax_mpl2.plot(
                            stock_data.index, stock_data['Close_MEP'], color=color, linewidth=1,
                            linestyle='-.', alpha=0.6, label=f'{display_name} (USD MEP)'
                        )

            if i == 0 and len(stock_data) > 0:
                stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index, y=stock_data['SMA'], mode='lines', name=f'{display_name} SMA (Ajustado)',
                        line=dict(color='orange', width=sma_line_width), yaxis='y1',
                        hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>SMA: %{{y:.2f}} {moneda}<extra></extra>'
                    )
                )

            y_top_ticker = (pct_change.max() if is_percentage_mode else stock_data['Inflation_Adjusted_Close'].max())

            splits_ticker = [
                (datetime.combine(s["date"], datetime.min.time()), f"Split {s['ratio']}:1")
                for s in st.session_state.custom_splits if s["ticker"] == ticker
            ]
            for dt, _ in splits_ticker:
                ax_mpl.axvline(dt, color='white', linewidth=1, linestyle='--', alpha=0.7)
            add_marker_lines(fig, splits_ticker, "white", y_top_ticker, f'{display_name} Splits', dash="dash")

            eventos_ticker = [
                (datetime.combine(e["date"], datetime.min.time()), e["description"])
                for e in st.session_state.custom_events if e["ticker"] == ticker
            ]
            for dt, _ in eventos_ticker:
                ax_mpl.axvline(dt, color='yellow', linewidth=1, linestyle=':', alpha=0.7)
            add_marker_lines(fig, eventos_ticker, "yellow", y_top_ticker, f'{display_name} Eventos')

        except Exception as e:
            st.error(f"Error procesando {ticker}: {e}")
            logger.error(f"Error processing {ticker}: {e}")
            continue

    fig.add_annotation(
        text="MTaurus - X: mtaurus_ok", xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=30, color="rgba(255, 255, 255, 0.2)"), opacity=0.15
    )

    tickers_titulo = ', '.join(tickers) if len(tickers) <= 3 else f"{len(tickers)} tickers"
    titulo_base = f'Precios Históricos Ajustados por Inflación ({moneda}) - {tickers_titulo}'
    titulo = titulo_base if not is_percentage_mode else f'{titulo_base} (%)'
    ylabel = f'Precio de Cierre Ajustado ({moneda})' if not is_percentage_mode else 'Variación Porcentual (%)'

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=20, color='white')),
        xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
        yaxis_title=dict(text=ylabel, font=dict(size=14, color='white')),
        **plot_style
    )
    fig.update_yaxes(
        type='log' if use_log_scale else 'linear',
        tickformat=',.2f',
        ticksuffix='' if not is_percentage_mode else '%'
    )
    if show_mep_ghost:
        fig.update_layout(
            yaxis2=dict(
                title=dict(text='Precio en USD (MEP)', font=dict(size=14, color='white')),
                overlaying='y', side='right', showgrid=False,
                tickformat=',.2f', color='white',
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    _finalizar_grafico_mpl(fig_mpl, ax_mpl, titulo, ylabel, is_percentage_mode, use_log_scale, ax2=ax_mpl2)
    st.pyplot(fig_mpl)
    plt.close(fig_mpl)

    return stock_data_dict_nominal, stock_data_dict_adjusted, ticker_var_map

# ------------------------------
# Diccionario de tickers y sus divisores
splits = {
    'ADGO.BA': {'ratio': 1, 'date': datetime(2024, 1, 23)},
    'ADBE.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'AEM.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'AMGN.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'AAPL.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'BAC.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'GOLD.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'BIOX.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'CVX.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'LLY.BA': {'ratio': 7, 'date': datetime(2024, 1, 23)},
    'XOM.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'FSLR.BA': {'ratio': 6, 'date': datetime(2024, 1, 23)},
    'IBM.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'JD.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'JPM.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'MELI.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'NFLX.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'PEP.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'PFE.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'PG.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'RIO.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'SONY.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'SBUX.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'TXR.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'BA.BA': {'ratio': 4, 'date': datetime(2024, 1, 23)},
    'TM.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'VZ.BA': {'ratio': 2, 'date': datetime(2024, 1, 23)},
    'VIST.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'WMT.BA': {'ratio': 3, 'date': datetime(2024, 1, 23)},
    'AGRO.BA': [
        {'ratio': 5.71, 'date': datetime(2023, 11, 2), 'type': 'divide'},
        {'ratio': 2.1, 'date': datetime(2023, 11, 6), 'type': 'multiply'},
    ],
    'ECOG.BA': {'ratio': 10, 'date': datetime(2025, 8, 18)},
}

# --- Redenominaciones de la moneda argentina (fechas fijas, históricas) ---
redenominations = [
    (datetime(1970, 1, 1), 2, 'Peso Ley 18.188'),
    (datetime(1983, 6, 1), 4, 'Peso Argentino'),
    (datetime(1985, 6, 15), 3, 'Austral'),
    (datetime(1992, 1, 1), 4, 'Peso'),
]

def get_currency(fecha):
    for change_date, _, currency in reversed(redenominations):
        if fecha >= change_date:
            return currency
    return 'Peso Moneda Nacional'

def to_current_peso(amount, fecha):
    """Convierte un monto de la moneda vigente en `fecha` a Pesos actuales
    (solo quita de ceros, sin inflación)."""
    for change_date, zeroes, _ in redenominations:
        if fecha < change_date:
            amount /= 10 ** zeroes
    return amount

def from_current_peso(amount, fecha):
    """Convierte Pesos actuales a la moneda vigente en `fecha` (sin inflación)."""
    for change_date, zeroes, _ in reversed(redenominations):
        if fecha < change_date:
            amount *= 10 ** zeroes
    return amount
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
    ticker_upper = ticker.upper()
    if "uploaded_data" in st.session_state and ticker_upper in st.session_state.uploaded_data:
        df = st.session_state.uploaded_data[ticker_upper]
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        return df
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
            if isinstance(adjustment, list):
                for paso in adjustment:
                    all_splits.append({
                        "date": paso["date"],
                        "ratio": paso["ratio"],
                        "type": paso.get("type", "divide")
                    })
            else:
                all_splits.append({
                    "date": adjustment["date"],
                    "ratio": adjustment["ratio"],
                    "type": adjustment.get("type", "divide")
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


# ------------------------------------------------------------------
# Ajuste por cupones cobrados (renta + amortización) de bonos
# ------------------------------------------------------------------
CLASES_EN_USD = {"Soberanos HD", "Bopreales"}


@st.cache_data(ttl=86400)
def cargar_cashflows_bonos():
    """
    Carga el cronograma de pagos de bonos (columnas esperadas: Bono, Fecha,
    Cashflow, Clase). Cashflow es el monto pagado por cada 100 de VN, ya en
    la moneda que corresponda según la Clase (USD para 'Soberanos HD' y
    'Bopreales'; ARS para el resto).
    """
    try:
        df = pd.read_csv('cashflows_bonos.csv', parse_dates=['Fecha'])
    except Exception:
        try:
            url = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/cashflows_bonos.csv"
            df = pd.read_csv(url, parse_dates=['Fecha'])
        except Exception as e:
            logger.warning(f"No se pudo cargar cashflows_bonos.csv ({e}). Ajuste por cupones deshabilitado.")
            st.sidebar.warning("⚠️ No se pudo cargar cashflows_bonos.csv: el ajuste por cupones de bonos está deshabilitado.")
            return pd.DataFrame()
    return df


@st.cache_data(ttl=86400)
def load_mep_data():
    """
    Serie diaria del dólar MEP (Bolsa), vía ArgentinaDatos/DolarApi.
    Se usa 'venta' como proxy del tipo de cambio de conversión de un cupón
    cobrado en USD; los días sin rueda (fines de semana/feriados) se
    completan con el último valor conocido.
    """
    try:
        response = requests.get("https://api.argentinadatos.com/v1/cotizaciones/dolares/bolsa", timeout=15)
        response.raise_for_status()
        mep = pd.DataFrame(response.json())
        mep["fecha"] = pd.to_datetime(mep["fecha"])
        mep = mep.rename(columns={"venta": "MEP"})[["fecha", "MEP"]].set_index("fecha").sort_index()
        rango_completo = pd.date_range(mep.index.min(), datetime.now().date(), freq="D")
        mep = mep.reindex(rango_completo).ffill()
        return mep["MEP"]
    except Exception as e:
        logger.error(f"Error obteniendo MEP histórico: {e}")
        return pd.Series(dtype=float)


def _resolver_ticker_bono(ticker, cashflows_df):
    """Normaliza el ticker ingresado (ej. 'AL30', 'AL30.BA') contra el
    ticker usado en el CSV de cashflows (ej. 'AL30D')."""
    base = ticker.upper().replace('.BA', '')
    tickers_csv = set(cashflows_df['Bono'].unique())
    if base in tickers_csv:
        return base
    if f"{base}D" in tickers_csv:
        return f"{base}D"
    if base.endswith('D') and base[:-1] in tickers_csv:
        return base[:-1]
    return None


def ajustar_precios_por_cupones(df, ticker, cashflows_df, mep_series):
    """
    Ajusta la serie a 'retorno total', simulando que cada cupón cobrado
    (renta + amortización) se reinvirtió en el mismo bono al precio
    vigente ese día. Evita que un pago se vea como una caída de precio.
    Si la clase del bono cobra en USD (Soberanos HD / Bopreales), el
    cupón se convierte a ARS con el MEP de la fecha de pago.
    Si el ticker no matchea ningún bono del CSV, devuelve `df` sin cambios.
    """
    try:
        if df.empty or cashflows_df is None or cashflows_df.empty:
            return df

        ticker_csv = _resolver_ticker_bono(ticker, cashflows_df)
        if ticker_csv is None:
            return df

        pagos = cashflows_df[cashflows_df['Bono'] == ticker_csv].sort_values('Fecha')
        if pagos.empty:
            return df

        cobra_en_usd = pagos['Clase'].iloc[0] in CLASES_EN_USD

        df = df.copy()
        precio_inicial_antes = df['Close'].iloc[0]
        factor = pd.Series(1.0, index=df.index)
        aplicados = 0
        sin_mep = 0

        for _, pago in pagos.iterrows():
            fecha_pago = pd.Timestamp(pago['Fecha'])
            if fecha_pago > df.index.max():
                continue  # el pago todavía no ocurrió dentro del rango mostrado

            # Por liquidación en 24hs, el bono cotiza ex-cupón desde la rueda
            # ANTERIOR a la fecha de pago del cronograma (no desde la fecha de
            # pago en sí). Esa rueda es donde realmente cae el precio.
            dias_previos = df.index[df.index < fecha_pago]
            if dias_previos.empty:
                continue  # no hay rueda anterior dentro del rango mostrado
            fecha_ex = dias_previos.max()

            monto = pago['Cashflow']
            if cobra_en_usd:
                if mep_series is None or mep_series.empty or fecha_pago not in mep_series.index:
                    sin_mep += 1
                    logger.warning(f"Sin MEP para {fecha_pago.date()}, se omite cupón de {ticker_csv} en esa fecha.")
                    continue
                monto = monto * mep_series.loc[fecha_pago]

            precio_ref = df.loc[fecha_ex, 'Close']
            if precio_ref <= 0:
                continue

            factor.loc[df.index < fecha_ex] *= precio_ref / (precio_ref + monto)
            aplicados += 1

        df['Close'] = df['Close'] * factor

        # Feedback visible para poder diagnosticar si el ajuste se está aplicando
        mensaje = (
            f"💰 {ticker} → cupones de **{ticker_csv}** "
            f"({'USD vía MEP' if cobra_en_usd else 'ARS'}): {aplicados} aplicado(s) en el rango mostrado"
        )
        if sin_mep:
            mensaje += f", {sin_mep} sin convertir por falta de MEP en esa fecha"
        mensaje += (
            f" | factor acumulado máx: {factor.max():.4f} | "
            f"precio en {df.index.min().date()}: {precio_inicial_antes:,.2f} → {df['Close'].iloc[0]:,.2f}"
        )
        st.caption(mensaje)

        return df

    except Exception as e:
        logger.error(f"Error ajustando cupones para {ticker}: {e}")
        st.warning(f"No se pudo ajustar por cupones para {ticker}: {e}")
        return df


def format_arg_amount(amount, decimals=2):
    if abs(amount) < 1e-6 and amount != 0:
        formatted_normal = f"{amount:,.12f}".replace(",", "X").replace(".", ",").replace("X", ".")
        formatted_scientific = f"{amount:.8e}".replace("e", "×10^")
        return formatted_normal, formatted_scientific
    formatted_normal = f"{amount:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted_normal, None


def amount_to_words(amount, currency, decimals=2):
    if abs(amount) < 1e-6 and amount != 0:
        formatted_normal, _ = format_arg_amount(amount, 12)
        return f"Valor muy pequeño: {formatted_normal} {currency}"

    entero = int(round(amount))
    decimales = int(round((amount - entero) * (10 ** decimals)))

    try:
        word_part = num2words(entero, lang='es').capitalize()
    except OverflowError:
        try:
            word_part = num2words(entero, lang='en').capitalize() + " (en inglés)"
        except OverflowError:
            formatted_normal, _ = format_arg_amount(amount, decimals)
            return f"Valor demasiado grande para expresar en palabras: {formatted_normal} {currency}"

    if decimales > 0:
        try:
            decimal_words = num2words(decimales, lang='es').capitalize()
            return f"{word_part} {currency} con {decimal_words} centavos"
        except OverflowError:
            decimal_words = num2words(decimales, lang='en').capitalize() + " (en inglés)"
            return f"{word_part} {currency} con {decimal_words} centavos"

    return f"{word_part} {currency}"
def _extrapolar_hasta_hoy(cpi_mensual: pd.DataFrame, meses_promedio: int = 3) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Recibe un DataFrame mensual con columna 'CPI_MoM' (tasa mensual, ej. 0.04 = 4%)
    indexado por fecha. Devuelve:
      - el mismo DataFrame con filas mensuales sintéticas agregadas hasta el mes actual,
        usando el promedio de la tasa de los últimos `meses_promedio` meses reales.
      - la fecha del último dato REAL (para mostrar el aviso).
    """
    cpi_mensual = cpi_mensual.sort_index()
    ultima_fecha_real = cpi_mensual.index.max()
    tasa_promedio = cpi_mensual['CPI_MoM'].tail(meses_promedio).mean()

    hoy = pd.Timestamp(datetime.now().date())
    if ultima_fecha_real >= hoy.to_period('M').to_timestamp():
        # Ya hay dato del mes actual, no hace falta extrapolar
        return cpi_mensual, ultima_fecha_real

    # Generar fechas mensuales sintéticas desde el mes siguiente al último real, hasta el mes actual
    fechas_sinteticas = pd.date_range(
        start=ultima_fecha_real + pd.offsets.MonthBegin(1),
        end=hoy,
        freq='MS'
    )
    if len(fechas_sinteticas) == 0:
        return cpi_mensual, ultima_fecha_real

    filas_sinteticas = pd.DataFrame(
        {'CPI_MoM': [tasa_promedio] * len(fechas_sinteticas)},
        index=fechas_sinteticas
    )
    cpi_extendido = pd.concat([cpi_mensual[['CPI_MoM']], filas_sinteticas])
    cpi_extendido = cpi_extendido[~cpi_extendido.index.duplicated(keep='first')]
    return cpi_extendido, ultima_fecha_real


def _construir_serie_diaria(cpi_mensual_extendido: pd.DataFrame) -> pd.Series:
    cpi = cpi_mensual_extendido.sort_index().copy()
    cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
    hoy = pd.Timestamp(datetime.now().date())
    # Aseguramos que la interpolación diaria llegue hasta hoy
    if cpi.index.max() < hoy:
        cpi.loc[hoy] = np.nan
        cpi = cpi.sort_index()
    daily = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear')
    daily = daily.ffill()  # por si el último tramo quedó NaN
    daily.index = pd.to_datetime(daily.index)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    return daily
# ------------------------------------------------------------------
# ARGENTINA - API Argentina Datos (INDEC)
# ------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_cpi_data():
    try:
        url = "https://api.argentinadatos.com/v1/finanzas/indices/inflacion"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        cpi_api = pd.DataFrame(data)
        cpi_api = cpi_api.rename(columns={"fecha": "Date", "valor": "CPI_MoM_pct"})
        cpi_api["Date"] = pd.to_datetime(cpi_api["Date"])
        cpi_api["CPI_MoM"] = cpi_api["CPI_MoM_pct"] / 100.0
        cpi_api.set_index("Date", inplace=True)
        cpi_api = cpi_api[["CPI_MoM"]]

        # 2007-2016: se prefiere el CSV curado por la manipulación histórica del INDEC.
        try:
            cpi_csv = pd.read_csv('inflaciónargentina2.csv')
            cpi_csv['Date'] = pd.to_datetime(cpi_csv['Date'], format='%d/%m/%Y')
            cpi_csv.set_index('Date', inplace=True)
            cpi_csv = cpi_csv[['CPI_MoM']]

            mask_indec = (cpi_api.index >= '2007-01-01') & (cpi_api.index <= '2016-12-31')
            cpi_api = cpi_api[~mask_indec]

            mask_csv = (cpi_csv.index >= '2007-01-01') & (cpi_csv.index <= '2016-12-31')
            cpi_curado = cpi_csv[mask_csv]

            cpi = pd.concat([cpi_api, cpi_curado]).sort_index()
            cpi = cpi[~cpi.index.duplicated(keep='last')]
        except Exception as e:
            logger.warning(f"No se pudo aplicar el CSV curado 2007-2016 ({e}). Usando solo API.")
            cpi = cpi_api

        cpi_extendido, ultima_fecha_real = _extrapolar_hasta_hoy(cpi)
        daily = _construir_serie_diaria(cpi_extendido)
        st.session_state['ipc_arg_ultima_fecha_real'] = ultima_fecha_real
        return daily
    except Exception as e:
        st.warning(f"No se pudo obtener el IPC de la API ({e}). Usando CSV local como respaldo.")
        return _load_cpi_data_csv_fallback()


def _load_cpi_data_csv_fallback():
    try:
        cpi = pd.read_csv('inflaciónargentina2.csv')
        cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
        cpi.set_index('Date', inplace=True)
        cpi_extendido, ultima_fecha_real = _extrapolar_hasta_hoy(cpi)
        daily = _construir_serie_diaria(cpi_extendido)
        st.session_state['ipc_arg_ultima_fecha_real'] = ultima_fecha_real
        return daily
    except Exception as e:
        st.error(f"Error loading CPI fallback CSV: {e}")
        st.stop()


# ------------------------------------------------------------------
# ESTADOS UNIDOS - FRED API (CPIAUCSL)
# ------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_us_cpi_data():
    try:
        api_key = os.environ.get("FRED_API_KEY") or st.secrets.get("FRED_API_KEY", None)
        if not api_key:
            raise ValueError("Falta FRED_API_KEY (variable de entorno o st.secrets).")
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": "CPIAUCSL", "api_key": api_key, "file_type": "json"}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()["observations"]
        cpi = pd.DataFrame(data)[["date", "value"]]
        cpi = cpi.rename(columns={"date": "Date", "value": "CPI_Level"})
        cpi["Date"] = pd.to_datetime(cpi["Date"])
        cpi["CPI_Level"] = pd.to_numeric(cpi["CPI_Level"], errors="coerce")
        cpi.dropna(subset=["CPI_Level"], inplace=True)
        cpi.set_index("Date", inplace=True)
        cpi.sort_index(inplace=True)
        cpi["CPI_MoM"] = cpi["CPI_Level"].pct_change()
        cpi.dropna(subset=["CPI_MoM"], inplace=True)
        cpi_api = cpi[["CPI_MoM"]]

        # La API (CPIAUCSL, desestacionalizada) arranca en 1947. Se completa con el
        # CSV para 1913-1946 (única serie oficial disponible para ese tramo; no está
        # desestacionalizada, pero no hay superposición de fechas con la API).
        try:
            url_csv = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nUSA.csv"
            cpi_csv = pd.read_csv(url_csv)
            cpi_csv['Date'] = pd.to_datetime(cpi_csv['Date'], format='%d/%m/%Y')
            cpi_csv.set_index('Date', inplace=True)
            cpi_csv = cpi_csv[['CPI_MoM']]

            primera_fecha_api = cpi_api.index.min()
            cpi_pre_api = cpi_csv[cpi_csv.index < primera_fecha_api]

            cpi = pd.concat([cpi_pre_api, cpi_api]).sort_index()
            cpi = cpi[~cpi.index.duplicated(keep='last')]
        except Exception as e:
            logger.warning(f"No se pudo completar el CPI de EE.UU. con el CSV pre-1947 ({e}). Usando solo API.")
            cpi = cpi_api

        cpi_extendido, ultima_fecha_real = _extrapolar_hasta_hoy(cpi)
        daily = _construir_serie_diaria(cpi_extendido)
        st.session_state['ipc_usa_ultima_fecha_real'] = ultima_fecha_real
        return daily
    except Exception as e:
        st.warning(f"No se pudo obtener el CPI de EE.UU. desde FRED ({e}). Usando CSV de respaldo.")
        return _load_us_cpi_data_csv_fallback()


def _load_us_cpi_data_csv_fallback():
    try:
        url = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nUSA.csv"
        cpi = pd.read_csv(url)
        cpi['Date'] = pd.to_datetime(cpi['Date'], format='%d/%m/%Y')
        cpi.set_index('Date', inplace=True)
        cpi_extendido, ultima_fecha_real = _extrapolar_hasta_hoy(cpi)
        daily = _construir_serie_diaria(cpi_extendido)
        st.session_state['ipc_usa_ultima_fecha_real'] = ultima_fecha_real
        return daily
    except Exception as e:
        st.error(f"Error loading US CPI fallback CSV: {e}")
        st.stop()
def mostrar_avisos_extrapolacion():
    """
    Llamar esto en el cuerpo principal del script, justo después de:
        daily_cpi = load_cpi_data()
        daily_us_cpi = load_us_cpi_data()
    """
    hoy = pd.Timestamp(datetime.now().date())
    fecha_arg = st.session_state.get('ipc_arg_ultima_fecha_real')
    fecha_usa = st.session_state.get('ipc_usa_ultima_fecha_real')

    if fecha_arg is not None and fecha_arg < hoy.to_period('M').to_timestamp():
        st.caption(
            f"⚠️ IPC Argentina: último dato oficial publicado es de "
            f"{fecha_arg.strftime('%B %Y')}. Los días posteriores usan una "
            f"estimación basada en el promedio de los últimos 3 meses."
        )
    if fecha_usa is not None and fecha_usa < hoy.to_period('M').to_timestamp():
        st.caption(
            f"⚠️ CPI EE.UU.: último dato oficial publicado es de "
            f"{fecha_usa.strftime('%B %Y')}. Los días posteriores usan una "
            f"estimación basada en el promedio de los últimos 3 meses."
        )

# Load CPI data
daily_cpi = load_cpi_data()
daily_us_cpi = load_us_cpi_data()
mostrar_avisos_extrapolacion()

# Load bond cashflows + MEP (para ajuste por cupones cobrados)
cashflows_bonos = cargar_cashflows_bonos()
daily_mep = load_mep_data()

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

st.sidebar.subheader("Paleta de colores de los gráficos")
paletas_disponibles = {
    'Clásico': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'Deep (Seaborn)': sns.color_palette('deep', 10).as_hex(),
    'Muted (Seaborn)': sns.color_palette('muted', 10).as_hex(),
    'Bright (Seaborn)': sns.color_palette('bright', 10).as_hex(),
    'Pastel (Seaborn)': sns.color_palette('pastel', 10).as_hex(),
    'Colorblind (Seaborn)': sns.color_palette('colorblind', 10).as_hex(),
    'Dark (Seaborn)': sns.color_palette('dark', 10).as_hex(),
}
paleta_seleccionada = st.sidebar.selectbox(
    "Elegí la paleta de colores para las líneas de los gráficos:",
    list(paletas_disponibles.keys()),
    key='paleta_colores_select'
)
colors = paletas_disponibles[paleta_seleccionada]

data_line_width = st.sidebar.slider(
    "Grosor de la línea de datos (precio/variación)",
    min_value=0.5, max_value=5.0, value=1.5, step=0.5,
    key="data_line_width",
)

sma_line_width = st.sidebar.slider(
    "Grosor de la línea SMA (solo Plotly)",
    min_value=0.5, max_value=5.0, value=1.5, step=0.5,
    key="sma_line_width",
)

st.sidebar.subheader("Cargar datos personalizados desde CSV")
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}
uploaded_files = st.sidebar.file_uploader("Subir archivos CSV", type=['csv'], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    filename = uploaded_file.name.lower()
    ticker = filename.replace('_d.csv', '').replace('.csv', '').upper()
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
        if 'Close' not in df.columns:
            st.sidebar.error(f"El CSV {uploaded_file.name} debe tener columna 'Close'.")
            continue
        st.session_state.uploaded_data[ticker] = df[['Close']]
        st.sidebar.success(f"Datos cargados para {ticker} desde {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error al cargar {uploaded_file.name}: {e}")

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

def _eventos_a_texto(eventos):
    return "\n".join(
        f"{e['ticker']},{e['date'].strftime('%Y-%m-%d')},{e['description']}"
        for e in eventos
    )

eventos_texto = st.sidebar.text_area(
    "Un evento por línea — formato: TICKER,AAAA-MM-DD,Descripción",
    value=_eventos_a_texto(st.session_state.custom_events),
    height=150,
    key="eventos_texto_input",
    help="Ejemplo:\nGGAL.BA,2024-03-15,Ganancias Q1\nYPFD.BA,2024-06-01,Anuncio dividendo",
)

if st.sidebar.button("Guardar eventos"):
    nuevos_eventos, errores = [], []
    for i, linea in enumerate(eventos_texto.splitlines(), start=1):
        linea = linea.strip()
        if not linea:
            continue
        partes = linea.split(",", 2)
        if len(partes) != 3:
            errores.append(f"Línea {i}: faltan comas (se esperan 3 campos)")
            continue
        ticker, fecha_str, descripcion = (p.strip() for p in partes)
        try:
            fecha = datetime.strptime(fecha_str, "%Y-%m-%d").date()
        except ValueError:
            errores.append(f"Línea {i}: fecha inválida '{fecha_str}' (usar AAAA-MM-DD)")
            continue
        if not ticker or not descripcion:
            errores.append(f"Línea {i}: ticker o descripción vacíos")
            continue
        nuevos_eventos.append({"ticker": ticker.upper(), "date": fecha, "description": descripcion})

    st.session_state.custom_events = nuevos_eventos
    if errores:
        st.sidebar.error("Algunas líneas no se cargaron:\n" + "\n".join(errores))
    else:
        st.sidebar.success(f"{len(nuevos_eventos)} eventos guardados.")

if st.session_state.custom_events:
    st.sidebar.caption(f"{len(st.session_state.custom_events)} eventos cargados.")
# Main content in tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Inflation Calculator", "Argentine Stock Adjuster", "Custom Calculations", "Volatility Analysis", "US Stock Adjuster"])

with tab1:
    st.subheader('Calculador de precios por inflación (Argentina)')

    st.markdown("""
    Esta calculadora te dice cuánto valdría **hoy** una plata que tenías en el pasado
    (o cuánto necesitabas en el pasado para comprar lo mismo que hoy).

    **Dos efectos separados que hay que tener en cuenta:**
    1. **Inflación**: con el tiempo, los precios suben y el dinero pierde poder de compra.
    2. **Cambios de moneda**: además de la inflación, Argentina cambió de moneda varias
       veces y le "sacó ceros" a los billetes para simplificarlos:
       - Peso Moneda Nacional (hasta 1970)
       - Peso Ley 18.188 (1970 en adelante, se sacaron 2 ceros)
       - Peso Argentino (1983 en adelante, se sacaron 4 ceros)
       - Austral (1985 en adelante, se sacaron 3 ceros)
       - Peso (1992 en adelante, se sacaron 4 ceros — es la moneda actual)

       Por ejemplo: $10.000.000 de Pesos Moneda Nacional (antes de 1970) equivalen,
       **solo por los cambios de moneda** (sin contar la inflación), a $1 Peso actual.
    """)

    value_choice = st.radio(
        "¿Qué querés calcular?",
        ('Fecha de Inicio', 'Fecha de Fin'),
        captions=[
            "Tengo un monto en el pasado y quiero saber cuánto vale hoy",
            "Tengo un monto de hoy y quiero saber cuánto necesitaba en el pasado"
        ],
        key='value_choice_radio'
    )

    incluir_cambios_moneda = st.checkbox(
        'Tener en cuenta los cambios de moneda (además de la inflación)',
        value=True,
        help=(
            "Si lo dejás tildado, además del ajuste por inflación te muestro el "
            "equivalente en Pesos de hoy, aplicando también la quita de ceros de "
            "cada cambio de moneda. Si lo destildás, solo ves el ajuste por "
            "inflación, en la moneda de esa época."
        ),
        key='incluir_cambios_moneda'
    )

    st.caption(
        "⚠️ Estos cálculos son aproximados: usan el IPC mensual interpolado día a día. "
        "Cuanto más largo el período o más alta la inflación acumulada, menos exacto "
        "es el resultado en el día a día (aunque el número final es confiable)."
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
            'Ingresa el monto que tenías en la fecha de inicio:',
            min_value=0.0,
            value=100.0,
            key='start_value_input'
        )

        start_dt = datetime.combine(start_date, datetime.min.time())
        moneda_inicio = get_currency(start_dt)

        try:
            start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
            end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
            factor = end_inflation / start_inflation
            end_value_misma_moneda = start_value * factor

            start_fmt, _ = format_arg_amount(start_value)
            end_fmt, end_fmt_sci = format_arg_amount(end_value_misma_moneda)

            st.markdown("#### Resultado")
            st.write(f"**Monto original:** {moneda_inicio} {start_fmt} (al {start_date.strftime('%d/%m/%Y')})")
            st.caption(amount_to_words(start_value, moneda_inicio))

            st.write(
                f"**Ajustado solo por inflación**, en la misma moneda de esa época "
                f"({moneda_inicio}): {moneda_inicio} {end_fmt}"
                + (f" ({end_fmt_sci})" if end_fmt_sci else "")
            )
            st.caption(
                f"Esto responde: ¿cuántos {moneda_inicio} necesitarías hoy, "
                f"**en esa misma moneda vieja**, para tener el mismo poder de compra? "
                "No tiene en cuenta que esa moneda ya no existe."
            )
            st.caption(amount_to_words(end_value_misma_moneda, moneda_inicio))

            if incluir_cambios_moneda:
                start_en_pesos_actuales = to_current_peso(start_value, start_dt)
                end_en_pesos_actuales = start_en_pesos_actuales * factor

                pesos_ini_fmt, pesos_ini_sci = format_arg_amount(start_en_pesos_actuales, 8)
                pesos_fin_fmt, pesos_fin_sci = format_arg_amount(end_en_pesos_actuales)

                st.write("---")
                st.write(
                    f"**Solo por el cambio de moneda** (sin inflación), esos "
                    f"{moneda_inicio} {start_fmt} equivalen hoy a: ARS {pesos_ini_fmt}"
                    + (f" ({pesos_ini_sci})" if pesos_ini_sci else "")
                )

                st.write(
                    f"**Resultado final (inflación + cambio de moneda), en Pesos actuales:** "
                    f"ARS {pesos_fin_fmt}" + (f" ({pesos_fin_sci})" if pesos_fin_sci else "")
                )
                st.caption(
                    "Este es el número más útil en la práctica: cuántos Pesos de hoy "
                    "necesitarías para tener el mismo poder de compra que tenías en la "
                    "fecha de inicio, contando todo (inflación y cambios de moneda)."
                )
                st.caption(amount_to_words(end_en_pesos_actuales, 'pesos'))
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
            'Ingresa el monto que tenés en la fecha de fin:',
            min_value=0.0,
            value=100.0,
            key='end_value_input'
        )

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())
        moneda_inicio = get_currency(start_dt)
        moneda_fin = get_currency(end_dt)

        try:
            start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
            end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
            factor = end_inflation / start_inflation
            start_value_misma_moneda = end_value / factor

            end_fmt, _ = format_arg_amount(end_value)
            start_fmt, start_fmt_sci = format_arg_amount(start_value_misma_moneda)

            st.markdown("#### Resultado")
            st.write(f"**Monto de referencia:** {moneda_fin} {end_fmt} (al {end_date.strftime('%d/%m/%Y')})")
            st.caption(amount_to_words(end_value, moneda_fin))

            st.write(
                f"**Deflactado solo por inflación**, en la misma moneda ({moneda_fin}): "
                f"{moneda_fin} {start_fmt}" + (f" ({start_fmt_sci})" if start_fmt_sci else "")
            )
            st.caption(
                f"Esto responde: ¿cuántos {moneda_fin} necesitabas en la fecha de "
                "inicio para tener el mismo poder de compra? Sin tener en cuenta "
                "que en esa época podía existir otra moneda."
            )
            st.caption(amount_to_words(start_value_misma_moneda, moneda_fin))

            if incluir_cambios_moneda:
                end_en_pesos_actuales = to_current_peso(end_value, end_dt)
                start_en_pesos_actuales = end_en_pesos_actuales / factor
                start_moneda_historica = from_current_peso(start_en_pesos_actuales, start_dt)

                pesos_fin_fmt, pesos_fin_sci = format_arg_amount(end_en_pesos_actuales)
                hist_fmt, hist_sci = format_arg_amount(start_moneda_historica, 8)

                st.write("---")
                st.write(
                    f"**Solo por el cambio de moneda**, ese monto equivale hoy a: "
                    f"ARS {pesos_fin_fmt}" + (f" ({pesos_fin_sci})" if pesos_fin_sci else "")
                )
                st.caption(amount_to_words(end_en_pesos_actuales, 'pesos'))

                st.write(
                    f"**Resultado final, en la moneda que circulaba el "
                    f"{start_date.strftime('%d/%m/%Y')} ({moneda_inicio}):** "
                    f"{moneda_inicio} {hist_fmt}" + (f" ({hist_sci})" if hist_sci else "")
                )
                st.caption(
                    f"Esto te dice cuántos billetes de {moneda_inicio} necesitabas en "
                    "esa fecha para comprar lo mismo que hoy — contando inflación y "
                    "cambios de moneda."
                )
                st.caption(amount_to_words(start_moneda_historica, moneda_inicio))
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

    plot_end_date = st.date_input(
        'Selecciona la fecha de fin para los datos mostrados en el gráfico:',
        min_value=plot_start_date,
        max_value=daily_cpi.index.max().date(),
        value=daily_cpi.index.max().date(),
        key='plot_end_date_input_arg'
    )

    force_inflation_arg = st.checkbox('Aplicar ajuste por inflación a todos los tickers (incluyendo no-.BA)', value=False, key='force_inflation_arg')

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

    show_nominal_ghost_arg = st.checkbox(
        'Incluir línea fantasma con el valor nominal (sin ajustar por inflación)',
        value=False,
        key='show_nominal_ghost_arg'
    )
    show_mep_ghost_arg = st.checkbox(
        'Incluir línea fantasma con el precio en USD MEP (eje secundario)',
        value=False,
        key='show_mep_ghost_arg'
    )

    # Diccionarios para almacenar datos (for Argentine tab)
    stock_data_dict_nominal_arg = {}
    stock_data_dict_adjusted_arg = {}

    if tickers_input:
        stock_data_dict_nominal_arg, stock_data_dict_adjusted_arg, ticker_var_map = graficar_activos_ajustados(
            tickers_input=tickers_input,
            sma_period=sma_period,
            plot_start_date=plot_start_date,
            daily_cpi_serie=daily_cpi,
            data_source=data_source,
            moneda='ARS',
            is_percentage_mode=is_percentage_mode,
            show_percentage_from_recent=show_percentage_from_recent,
            use_log_scale=use_log_scale_arg,
            show_nominal_ghost=show_nominal_ghost_arg,
            siempre_ajustar=False,
            force_inflation=force_inflation_arg,
            plot_end_date=plot_end_date,
            sma_line_width=sma_line_width,
            data_line_width=data_line_width,
            show_mep_ghost=show_mep_ghost_arg,
        )

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

    # ←←← NUEVO CAMPO PARA TÍTULO PERSONALIZADO ←←←
    custom_title = st.text_input(
        'Título personalizado del gráfico (opcional):',
        placeholder='Dejar vacío para título automático',
        key='custom_title_input'
    )

    if custom_expression:
        try:
            # ------------------------------------------------------------------
            # 1. Detectar tickers usados en la expresión
            # ------------------------------------------------------------------
            sorted_tickers = sorted(ticker_var_map.keys(), key=len, reverse=True)
            transformed_expression = custom_expression
            used_tickers = set()
            used_ba_tickers = set()

            for ticker in sorted_tickers:
                if ticker in custom_expression:
                    used_tickers.add(ticker)
                    var_name = ticker_var_map[ticker]
                    pattern = re.escape(ticker)
                    transformed_expression = re.sub(rf'\b{pattern}\b', var_name, transformed_expression)
                    if ticker.endswith('.BA'):
                        used_ba_tickers.add(ticker)

            # ------------------------------------------------------------------
            # 2. DataFrame combinado
            # ------------------------------------------------------------------
            combined_nominal_df = pd.DataFrame({
                ticker_var_map[ticker]: stock_data_dict_nominal_arg[ticker_var_map[ticker]]
                for ticker in used_tickers
            })
            combined_nominal_df.dropna(inplace=True)

            if combined_nominal_df.empty:
                st.error("No hay datos disponibles para todos los tickers seleccionados en las fechas especificadas.")
            else:
                # ------------------------------------------------------------------
                # 3. Evaluar expresión
                # ------------------------------------------------------------------
                custom_series_nominal = combined_nominal_df.eval(transformed_expression, engine='python')
                custom_series_nominal = custom_series_nominal.to_frame(name='Custom_Nominal')

                # ------------------------------------------------------------------
                # 4. Ajuste por inflación si hay algún .BA
                # ------------------------------------------------------------------
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

                # ------------------------------------------------------------------
                # 5. Gráfico
                # ------------------------------------------------------------------
                fig = go.Figure()

                # ── Traza principal ──
                if show_percentage or show_percentage_from_recent:
                    if show_percentage_from_recent:
                        custom_series_pct = (adjusted_series / adjusted_series.iloc[-1] - 1) * 100
                        custom_series_pct = -custom_series_pct
                    else:
                        custom_series_pct = (adjusted_series / adjusted_series.iloc[0] - 1) * 100

                    fig.add_trace(go.Scatter(
                        x=custom_series_pct.index,
                        y=custom_series_pct,
                        mode='lines',
                        name=f'Custom: {custom_expression[:15]}...' if len(custom_expression) > 15 else custom_expression,
                        line=dict(color=colors[-1], width=2),
                        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación: %{y:.2f}%<extra></extra>'
                    ))
                    fig.add_hline(y=0, line=dict(color="rgba(255,0,0,0.5)", dash="dash"))
                else:
                    fig.add_trace(go.Scatter(
                        x=adjusted_series.index,
                        y=adjusted_series,
                        mode='lines',
                        name=f'Custom: {custom_expression[:15]}...' if len(custom_expression) > 15 else custom_expression,
                        line=dict(color=colors[-1], width=2),
                        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Valor: %{y:.2f} ARS<extra></extra>'
                    ))

                # ── Eventos y splits (líneas sin texto fijo, info al pasar el mouse) ──
                serie_top = custom_series_pct if (show_percentage or show_percentage_from_recent) else adjusted_series
                y_top_custom = serie_top.max()

                eventos_custom = [
                    (datetime.combine(e["date"], datetime.min.time()), e["description"])
                    for e in st.session_state.get("custom_events", []) if e["ticker"] in used_tickers
                ]
                add_marker_lines(fig, eventos_custom, "yellow", y_top_custom, "Eventos")

                splits_custom = [
                    (datetime.combine(s["date"], datetime.min.time()), f"Split {s['ratio']}:1")
                    for s in st.session_state.get("custom_splits", []) if s["ticker"] in used_tickers
                ]
                add_marker_lines(fig, splits_custom, "white", y_top_custom, "Splits", dash="dash")

                # ── Título (personalizado o automático) ──
                if custom_title.strip():
                    plot_title = custom_title.strip()
                else:
                    if show_percentage or show_percentage_from_recent:
                        plot_title = 'Ratio / Cálculo Personalizado (%)'
                    else:
                        plot_title = 'Ratio / Cálculo Personalizado Ajustado por Inflación'

                fig.update_layout(
                    title=dict(text=plot_title, font=dict(size=20, color='white')),
                    xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
                    yaxis_title=dict(
                        text='Variación (%)' if (show_percentage or show_percentage_from_recent) else 'Valor Ajustado (ARS)',
                        font=dict(size=14, color='white')
                    ),
                    **plot_style
                )

                fig.update_yaxes(
                    type='log' if (not (show_percentage or show_percentage_from_recent) and use_log_scale_arg) else 'linear',
                    tickformat=',.2f',
                    ticksuffix='%' if (show_percentage or show_percentage_from_recent) else ''
                )

                # Watermark
                fig.add_annotation(
                    text="MTaurus - X: mtaurus_ok",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=30, color="rgba(255, 255, 255, 0.15)"),
                    opacity=0.2
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            available_vars = ', '.join(ticker_var_map.values())
            st.error(f"Error al evaluar la expresión: {e}\n\nVariables disponibles: {available_vars}")

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
                stock_data = ajustar_precios_por_cupones(stock_data, ticker, cashflows_bonos, daily_mep)

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

                splits_vol = [
                    (datetime.combine(s["date"], datetime.min.time()), f"Split {s['ratio']}:1")
                    for s in st.session_state.custom_splits if s["ticker"] == ticker
                ]
                add_marker_lines(
                    fig_vol, splits_vol, "white",
                    stock_data['Inflation_Adjusted_Close'].max(), f'{display_name} Splits', dash="dash"
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

    plot_end_date_us = st.date_input(
        'Selecciona la fecha de fin para los datos mostrados en el gráfico:',
        min_value=plot_start_date_us,
        max_value=daily_us_cpi.index.max().date(),
        value=daily_us_cpi.index.max().date(),
        key='plot_end_date_input_us'
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

    show_nominal_ghost_us = st.checkbox(
        'Incluir línea fantasma con el valor nominal (sin ajustar por inflación)',
        value=False,
        key='show_nominal_ghost_us'
    )

    # Diccionarios para almacenar datos (for US tab)
    stock_data_dict_nominal_us = {}
    stock_data_dict_adjusted_us = {}

    if tickers_input_us:
        stock_data_dict_nominal_us, stock_data_dict_adjusted_us, ticker_var_map_us = graficar_activos_ajustados(
            tickers_input=tickers_input_us,
            sma_period=sma_period_us,
            plot_start_date=plot_start_date_us,
            daily_cpi_serie=daily_us_cpi,
            data_source=data_source,
            moneda='USD',
            is_percentage_mode=is_percentage_mode_us,
            show_percentage_from_recent=show_percentage_from_recent_us,
            use_log_scale=use_log_scale_us,
            show_nominal_ghost=show_nominal_ghost_us,
            siempre_ajustar=True,
            plot_end_date=plot_end_date_us,
            sma_line_width=sma_line_width,
            data_line_width=data_line_width,
        )
