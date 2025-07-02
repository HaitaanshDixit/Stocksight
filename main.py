import os
import time
import json
import joblib
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
from pyngrok import ngrok

from src.utils import plot_graph, preprocess_stock_data_lstm

st.set_page_config(page_title="Stocksight", layout="centered")
st.title("Stocksight - Stock Price Predictor")

# Inputs
stock = st.selectbox("Choose Stock", ["RELIANCE.NS"], index=None, placeholder="Click to choose stock")
next_days = st.number_input("Enter number of future days to predict (max 15)", min_value=1, step=1, placeholder="Enter here")

if next_days > 15:
    st.warning("Please enter a value less than or equal to 15.")
    st.stop()

# Asset Map 
stock_assets = {
    "RELIANCE.NS": {
        "model_file": "saved_models/Reliance_model.keras",
        "scaler_file": "saved_models/scaler.save",
        "metrics_file": "model_metrics/Reliance_model_metrics.json"
    }
}

# Execution 
if stock is None:
    st.info("ℹ️ Please select a stock to continue.")
elif stock in stock_assets:
    st.markdown("---")
    st.subheader(f"Model & Forecast for {stock}")

    if st.button("Predict here"):
        # Load model and scaler
        model = load_model(stock_assets[stock]["model_file"])
        scaler = joblib.load(stock_assets[stock]["scaler_file"])

        # Display metrics
        st.write("")
        st.markdown("Model Evaluation Metrics")
        st.markdown("These metrics were collected during training & validation of the LSTM model.")
        with open(stock_assets[stock]["metrics_file"], "r") as f:
            metrics = json.load(f)
        st.json(metrics)

        st.markdown("---")

        # Load stock data
        df = yf.download(stock, period="180d")
        df_clean = df[['Open', 'High', 'Close']]

        st.subheader("Recent Stock Data (Last 180 Days)")
        st.dataframe(df_clean.tail())

        st.markdown("---")

        # EDA
        st.subheader("Price Trend, Seasonality & PACF (10 Years)")
        dff = yf.download(stock, period="10y", interval="1d")
        dff_cl = dff[['Open', 'High', 'Close']]
        plot_graph(dff_cl)

        st.markdown("---")

        # Prepare for prediction
        X_input, _, _ = preprocess_stock_data_lstm(df_clean, lags=30, scaler=scaler, is_train=False)
        predictions = []
        current_input = X_input[-1:].copy()

        for _ in range(next_days):
            next_pred = model.predict(current_input)[0][0]
            predictions.append(next_pred)
            next_input = np.append(current_input[:, 1:, :], [[[next_pred]*3]], axis=1)
            current_input = next_input

        predicted_prices = scaler.inverse_transform([[p]*3 for p in predictions])[:, 0]

        # Latest price
        latest_price = df['Close'].iloc[-1].item()
        st.subheader(f"Latest Available Closing Price for {stock} : ₹{latest_price:.2f}")

        st.markdown("---")

        # Forecast Plot
        st.subheader(f"{next_days}-Day Price Forecast for {stock}")
        future_dates = pd.date_range(start=df.index[-1], periods=next_days + 1, freq='B')[1:]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predicted_prices})
        forecast_df.set_index('Date', inplace=True)
        st.line_chart(forecast_df)

        # Download forecast as CSV
        st.download_button(
            label="Download the Forecast here",
            data=forecast_df.to_csv().encode('utf-8'),
            file_name=f"{stock}_forecast.csv",
            mime='text/csv'
        )

else:
    st.warning("🚧 This stock is not yet supported.")

#---------------------------------------------------------------------------------------------------  

ngrok.kill()
time.sleep(2)
ngrok.set_auth_token("2z9A28wqC8QpClwA08VkvAvOzsL_5CVDfosVWkQGfsLdajzw7")
public_url = ngrok.connect(8501)
print(f"🔗 Public URL: {public_url}")

os.system("streamlit run main.py --server.port 8501 > /dev/null 2>&1 &")
