import yfinance as yf
import pandas as pd
import schedule
import time

def fetch_and_save_data():
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB", "TSLA", "NVDA", "JPM", "BAC", "WFC"]
    historical_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info is not None:
                historical = yf.download(ticker, period='30d')
                historical['Ticker'] = ticker
                historical_data = pd.concat([historical_data, historical])
            else:
                print(f"Skipping {ticker}: Data not available")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    if not historical_data.empty:
        historical_data.to_csv('all_historical_data.csv', index=True)
        
        # Save info data
        info_df = pd.DataFrame()
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                info_df = pd.concat([info_df, pd.DataFrame([info])], ignore_index=True)
            except Exception as e:
                print(f"Error fetching info for {ticker}: {e}")
        
        info_df.to_csv('stock_info.csv', index=False)
    else:
        print("No valid data to save.")

def start_scheduler():
    fetch_and_save_data()
    schedule.every(30).minutes.do(fetch_and_save_data)  # Fetch every 30 minutes

if __name__ == "__main__":
    start_scheduler()
    while True:
        schedule.run_pending()
        time.sleep(1)
