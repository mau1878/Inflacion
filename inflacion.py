# Add a new checkbox for the alternative percentage calculation
show_percentage_from_recent = st.checkbox('Mostrar valores ajustados por inflación como porcentajes desde el valor más reciente', value=False)

# Modify the existing logic to include the new option
if show_percentage or show_percentage_from_recent:
  if show_percentage_from_recent:
      # Calculate percentages using the most recent value as the reference
      stock_data['Inflation_Adjusted_Percentage'] = (stock_data['Inflation_Adjusted_Close'] / stock_data['Inflation_Adjusted_Close'].iloc[-1] - 1) * 100
  else:
      # Calculate percentages using the initial value as the reference
      stock_data['Inflation_Adjusted_Percentage'] = (stock_data['Inflation_Adjusted_Close'] / stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100

  # Plot the percentage changes
  fig.add_trace(go.Scatter(
      x=stock_data.index,
      y=stock_data['Inflation_Adjusted_Percentage'],
      mode='lines',
      name=f'{ticker} (%)',
      yaxis='y1'
  ))

  # Add a horizontal red line at 0%
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
  # Plot the inflation-adjusted prices
  fig.add_trace(go.Scatter(
      x=stock_data.index,
      y=stock_data['Inflation_Adjusted_Close'],
      mode='lines',
      name=f'{ticker}',
      yaxis='y1'
  ))

  # Plot the average price as a dotted line
  avg_price = stock_data['Inflation_Adjusted_Close'].mean()
  fig.add_trace(go.Scatter(
      x=stock_data.index,
      y=[avg_price] * len(stock_data),
      mode='lines',
      name=f'{ticker} Precio Promedio',
      line=dict(dash='dot'),
      yaxis='y1'
  ))
