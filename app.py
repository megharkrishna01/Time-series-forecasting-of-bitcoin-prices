from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', plot_div=None, predicted_values=None, test_rmse=None)

@app.route('/predict', methods=['POST'])
def predict():
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    data = yf.download(tickers='BTC-USD', start='2019-01-01', end='2023-06-24', interval='1d')
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    df = data[['Date', 'Open', 'High', 'Low', 'Close']].copy()
    df.columns = ['ds', 'Open', 'High', 'Low', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    df_train = df[df['ds'] <= '2023-05-31']
    df_test = df[df['ds'] > '2023-05-31']

    m = Prophet(interval_width=0.80, n_changepoints=40 )
    m.fit(df_train)

    future = m.make_future_dataframe(periods=40)  # Increase periods to 40 for 20 additional days
    future = future[(future['ds'] >= start_date) & (future['ds'] <= end_date)]  # Filter future dates based on user input
    forecast = m.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['ds'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['y'],
                                 name='Bitcoin Data'))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Close'))
    fig.update_layout(xaxis_rangeslider_visible=False)
    plot_div = fig.to_html(full_html=False)

    # Get the predicted values as a list of dictionaries for the filtered range
    predicted_values = forecast[['ds', 'yhat']].values.tolist()

    # Evaluate the model performance on the test set
    test_predictions = m.predict(df_test)
    test_rmse = np.sqrt(np.mean((test_predictions['yhat'].values - df_test['y'].values) ** 2))

    return render_template('index.html', plot_div=plot_div, predicted_values=predicted_values, test_rmse=test_rmse)

if __name__ == '__main__':
    app.run(debug=True)
