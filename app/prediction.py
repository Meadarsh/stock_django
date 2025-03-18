import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from plotly.offline import plot
import plotly.graph_objects as go
from datetime import timedelta

def fetch_and_predict(ticker_value, number_of_days):
    try:
        # Fetch stock data using yfinance
        df = yf.download(ticker_value, start="2019-01-01", end="2024-12-31", progress=False)
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker_value}'.")

        # Data Preprocessing
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare LSTM sequences
        def create_sequences(data, seq_length):
            xs = []
            for i in range(len(data) - seq_length):
                x = data[i:i + seq_length]
                xs.append(x)
            return np.array(xs)

        SEQ_LENGTH = 60  # Sequence length for LSTM
        X_test = create_sequences(scaled_data, SEQ_LENGTH)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Load pre-trained LSTM model
        model = load_model("app/Data/lstm_model.h5")

        # Make predictions on historical data
        y_predicted_scaled = model.predict(X_test)
        y_predicted = scaler.inverse_transform(y_predicted_scaled)

        # Future Predictions (Next `number_of_days`)
        def generate_future_predictions(model, last_sequence, num_days, noise_factor=0.02):
            future_predictions = []
            current_sequence = last_sequence.copy()

            for _ in range(num_days):
                prediction_scaled = model.predict(current_sequence.reshape(1, SEQ_LENGTH, 1))[0][0]
                
                # Add noise to the prediction
                noise = np.random.normal(0, noise_factor)
                prediction_scaled += noise
                prediction_scaled = max(0, min(1, prediction_scaled))  # Clip to [0, 1]

                future_predictions.append(prediction_scaled)
                current_sequence = np.append(current_sequence[1:], [[prediction_scaled]], axis=0)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            return future_predictions

        last_sequence = scaled_data[-SEQ_LENGTH:]
        future_prices = generate_future_predictions(model, last_sequence, int(number_of_days))

        # Generate dates for future predictions
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, int(number_of_days) + 1)]

        # Create a DataFrame for future predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices.flatten()
        })
        future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')  # Format dates

        # Convert DataFrame to list of dictionaries for the template
        future_data = future_df.to_dict('records')

        # Plot Historical and Future Predictions
        fig = go.Figure()

        # Historical Prices
        fig.add_trace(go.Scatter(x=df.index[-len(y_predicted):], y=y_predicted.flatten(),
                                 mode='lines', name='Predicted Prices'))

        # Future Predictions
        fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Price'],
                                 mode='lines+markers', name='Future Predictions'))

        fig.update_layout(title=f"Stock Price Prediction for {ticker_value}",
                          xaxis_title="Date", yaxis_title="Price",
                          paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

        plot_div = plot(fig, output_type='div')

        return plot_div, future_data

    except Exception as e:
        return f"Error: {str(e)}", None
