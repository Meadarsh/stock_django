from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objects as go
import pandas as pd
import json
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
from .prediction import fetch_and_predict

# The Home page when Server loads up
def index(request):
    # ================== Left Card Plot ==================
    data = yf.download(
        tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM'],
        group_by='ticker',
        period='1mo',
        interval='1d',
        auto_adjust=False  # Ensure consistent columns
    ).reset_index()

    fig_left = go.Figure()
    for ticker in ['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM']:
        fig_left.add_trace(
            go.Scatter(x=data['Date'], y=data[ticker]['Close'], name=ticker)
        )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_left = plot(fig_left, output_type='div')

    # ================== Recent Stocks ==================
    def get_recent(ticker):
        df = yf.download(ticker, period='1d', interval='1d')
        df.insert(0, "Ticker", ticker)
        return df[['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].reset_index()

    tickers = ['AAPL', 'AMZN', 'GOOGL', 'UBER', 'TSLA', 'TWTR']
    df = pd.concat([get_recent(t) for t in tickers]).drop('Date', axis=1)
    recent_stocks = json.loads(df.to_json(orient='records'))

    return render(request, 'app/index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })


def search(request):
    return render(request, 'app/search.html', {})


# The Predict Function to handle predictions and plotting
def predict(request, ticker_value, number_of_days):
    try:
        plot_div, future_data = fetch_and_predict(ticker_value, number_of_days)
        return render(request, 'app/result.html', {'plot_div': plot_div, 'future_data': future_data})
    except Exception as e:
        return render(request, 'app/error.html', {'error': str(e)})

def ticker(request, ticker_value):
    try:
        stock = yf.Ticker(ticker_value)
        info = stock.info

        if not info:
            raise ValueError(f"Could not retrieve data from yfinance for ticker '{ticker_value}'.")

        # Prepare the data for the template
        data = {
            "ticker": ticker_value,
            "companyName": info.get('longName', 'N/A'),
            "currentPrice": info.get('currentPrice', 'N/A'),
            "previousClose": info.get('previousClose', 'N/A'),
            "marketCap": info.get('marketCap', 'N/A'),
            "volume": info.get('volume', 'N/A'),
            # Add any other relevant data you want to display
        }

        return render(request, 'app/ticker.html', {'data': data})

    except Exception as e:
        return render(request, 'app/error.html', {'error': str(e)})
