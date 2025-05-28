import pandas as pd
import numpy as np
import datetime

def generate_financial_data(
    days: int = 500,
    initial_price: float = 100.0,
    trend_slope: float = 0.05,
    seasonality_amplitude: float = 10.0,
    seasonality_period: int = 90,
    noise_level: float = 2.0,
    base_volume: int = 100000,
    volume_volatility: float = 0.3,
    start_date_str: str = "2022-01-01"
) -> pd.DataFrame:
    """
    Generates synthetic time series data resembling daily stock prices.

    Args:
        days (int): Number of trading days to generate data for.
        initial_price (float): Starting price for the stock.
        trend_slope (float): Slope for the linear trend component (price change per day).
        seasonality_amplitude (float): Amplitude of the seasonal (sine wave) component.
        seasonality_period (int): Period in days for the seasonality.
        noise_level (float): Standard deviation of the random noise added to the price.
        base_volume (int): Base daily trading volume.
        volume_volatility (float): Percentage volatility for volume (e.g., 0.3 for +/-30%).
        start_date_str (str): Start date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Date', 'Close', 'Volume'].
                      'Date' is datetime, 'Close' is float, 'Volume' is int.
    """

    # 1. Create date range
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date_str must be in 'YYYY-MM-DD' format")
    
    # Using pandas date_range for business days if needed, or simple days
    # For simplicity, using consecutive days. For actual trading days, use bdate_range.
    dates = pd.date_range(start_date, periods=days, freq='D')
    
    # Time component for trend and seasonality
    time_component = np.arange(days)

    # 2. Generate linear trend component
    trend = trend_slope * time_component

    # 3. Generate seasonal component (sine wave)
    seasonal = seasonality_amplitude * np.sin(2 * np.pi * time_component / seasonality_period)

    # 4. Generate random noise
    noise = np.random.normal(loc=0, scale=noise_level, size=days)

    # 5. Calculate Close price
    close_prices = initial_price + trend + seasonal + noise
    # Ensure prices don't go below a certain minimum (e.g., 1.0)
    close_prices = np.maximum(close_prices, 1.0) 

    # 6. Generate Volume data
    volume_random_factor = np.random.uniform(-volume_volatility, volume_volatility, size=days)
    volumes = base_volume * (1 + volume_random_factor)
    # Ensure volume is positive and integer
    volumes = np.maximum(volumes, 0).astype(int)

    # 7. Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': close_prices,
        'Volume': volumes
    })

    # For OHLC (Optional - can be expanded later)
    # Open: Could be previous day's close, or close +/- small random factor
    # High: Close + small positive random factor
    # Low: Close - small positive random factor (ensure Low <= Open/Close and High >= Open/Close)
    # For now, these are omitted as per requirements.

    return df

if __name__ == "__main__":
    # Generate data with default parameters
    print("Generating synthetic financial data with default parameters...")
    financial_df = generate_financial_data()

    # Print head and tail
    print("\nDataFrame Head:")
    print(financial_df.head())
    print("\nDataFrame Tail:")
    print(financial_df.tail())
    print(f"\nGenerated {len(financial_df)} days of data.")

    # Save to CSV
    output_filename = "synthetic_financial_data.csv"
    try:
        financial_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved data to {output_filename}")
    except Exception as e:
        print(f"\nError saving data to CSV: {e}")

    # Example with custom parameters
    print("\nGenerating synthetic financial data with custom parameters...")
    custom_financial_df = generate_financial_data(
        days=100,
        initial_price=50.0,
        trend_slope=-0.1, # Downward trend
        seasonality_amplitude=5.0,
        seasonality_period=30,
        noise_level=1.0,
        base_volume=50000,
        start_date_str="2023-01-01"
    )
    print("\nCustom DataFrame Head:")
    print(custom_financial_df.head())
    custom_output_filename = "custom_synthetic_financial_data.csv"
    try:
        custom_financial_df.to_csv(custom_output_filename, index=False)
        print(f"\nSuccessfully saved custom data to {custom_output_filename}")
    except Exception as e:
        print(f"\nError saving custom data to CSV: {e}")
