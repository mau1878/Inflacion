# User input: select the start date for the data shown in the plot
plot_start_date = st.date_input(
    'Selecciona la fecha de inicio para los datos mostrados en el gráfico:',
    min_value=daily_cpi.index.min().date(),
    max_value=daily_cpi.index.max().date(),
    value=daily_cpi.index.min().date(),
    key='plot_start_date_input'
)

# User input: select the "zero-percent date"
zero_percent_date = st.date_input(
    'Selecciona la fecha que se utilizará como referencia de 0%:',
    min_value=plot_start_date,
    max_value=daily_cpi.index.max().date(),
    value=daily_cpi.index.max().date(),
    key='zero_percent_date_input'
)

if tickers_input:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        try:
            # Fetch historical stock data starting from the user-selected plot start date
            stock_data = yf.download(ticker, start=plot_start_date, end=daily_cpi.index.max().date())
            
            if stock_data.empty:
                st.error(f"No se encontraron datos para el ticker {ticker}.")
                continue

            # Ensure the Date column is in datetime format
            stock_data.index = pd.to_datetime(stock_data.index)

            # Adjust stock prices for inflation
            stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (daily_cpi.loc[stock_data.index[-1]] / daily_cpi.loc[stock_data.index])

            # Calculate the zero-percent reference price
            zero_percent_price = stock_data.loc[pd.to_datetime(zero_percent_date), 'Inflation_Adjusted_Close']

            # Calculate performance relative to the zero-percent date
            stock_data['Performance'] = ((stock_data['Inflation_Adjusted_Close'] / zero_percent_price) - 1) * 100

            # Plot the performance
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Performance'],
                                     mode='lines', name=ticker))

            # Plot the SMA for the first ticker only
            if i == 0:
                stock_data['SMA'] = stock_data['Performance'].rolling(window=sma_period).mean()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'],
                                         mode='lines', name=f'{ticker} SMA de {sma_period} Periodos',
                                         line=dict(color='orange')))
        
        except Exception as e:
            st.error(f"Ocurrió un error con el ticker {ticker}: {e}")

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
    fig.update_layout(title='Rendimiento Ajustado por Inflación (Referencia 0% en la Fecha Seleccionada)',
                      xaxis_title='Fecha',
                      yaxis_title='Rendimiento (%)')

    st.plotly_chart(fig)
