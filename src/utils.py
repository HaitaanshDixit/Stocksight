from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
import streamlit as st


def plot_graph(df_clean):
    # Closing prices for 8 years
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_clean.index, df_clean['Close'], color='blue')
    ax1.set_title("Stock Prices of Last 10 Years")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Closing Prices")
    st.pyplot(fig1)

    # Trend, Seasonality, Residual Plot
    close_series = df_clean["Close"].dropna()

    # Only decompose if enough data is available
    if len(close_series) >= 504:
        decomposition = seasonal_decompose(close_series, model='multiplicative', period=252)

        fig2, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(decomposition.trend, color='black')
        axs[0].set_title("Trend")

        axs[1].plot(decomposition.seasonal, color='orange')
        axs[1].set_title("Seasonality")

        axs[2].plot(decomposition.resid, color='red')
        axs[2].set_title("Residual (Noise)")

        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning("❗ Not enough data points for seasonal decomposition (min 504 required).")

    # PACF Plot
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    plot_pacf(close_series, lags=30, method='ywm', ax=ax3)
    ax3.set_title("Partial Autocorrelation Function (PACF)")
    st.pyplot(fig3)

    # 20-day moving average
    df_clean['MA_20'] = df_clean['Close'].rolling(window=20).mean()

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(df_clean['Close'], label='Close Price', linewidth=1.5)
    ax4.plot(df_clean['MA_20'], label='20-Day Moving Average', color='orange', linewidth=2)
    ax4.set_title('Close Price vs 20-Day Moving Average')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.grid(True)
    plt.tight_layout()
    st.pyplot(fig4)

# training data, validation data and test data

# We will try to change the ratios if accuaracy not satisfactory for model training.
import pandas as pd
import numpy as np

def split_df_clean(df_clean, train_ratio=0.875, val_ratio=0.0625):

    df_clean = df_clean[['Open', 'High', 'Close']].dropna().copy()

    total_len = len(df_clean)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    df_train = df_clean.iloc[:train_end]
    df_val   = df_clean.iloc[train_end:val_end]
    df_test  = df_clean.iloc[val_end:]

    return df_train, df_val, df_test

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_stock_data_lstm(df_split, lags=30, scaler=None, is_train=True):
    df_split = df_split[['Open', 'High', 'Close']].dropna().copy()

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df_split)
    else:
        scaled_values = scaler.transform(df_split)

    df_scaled = pd.DataFrame(scaled_values, columns=['Open', 'High', 'Close'], index=df_split.index)

    X, y = [], []
    for i in range(lags, len(df_scaled)):
        X.append(df_scaled.iloc[i-lags:i].values)
        y.append(df_scaled.iloc[i]['Close'])

    return np.array(X), np.array(y), scaler

