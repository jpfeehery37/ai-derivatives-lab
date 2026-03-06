# Exploratory: fetch SPY data with yfinance.
import yfinance as yf

df = yf.download("SPY", period="5d")
print(df)
