import torch
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorus.tensor_storage import TensorStorage # Assuming direct import works
import typing
from typing import List, Dict, Any, Optional, Tuple
import logging
import time # For timestamp in metadata if needed

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

def load_time_series_from_tensorus(
    storage: TensorStorage,
    dataset_name: str,
    series_metadata_field: str = "name",
    series_name: str = "dummy_stock_close",
    date_metadata_field: Optional[str] = "dates"
) -> Optional[pd.Series]:
    """
    Loads a time series from TensorStorage.

    Args:
        storage: Instance of TensorStorage.
        dataset_name: Name of the dataset in TensorStorage.
        series_metadata_field: Metadata field used to identify the time series.
        series_name: Specific value of series_metadata_field to filter for.
        date_metadata_field: Optional metadata field containing dates for the series.

    Returns:
        A Pandas Series with a DatetimeIndex or RangeIndex, or None if not found/error.
    """
    logging.info(f"Attempting to load series '{series_name}' from dataset '{dataset_name}' using metadata field '{series_metadata_field}'.")
    try:
        if dataset_name not in storage.list_datasets():
            logging.error(f"Dataset '{dataset_name}' not found in TensorStorage.")
            return None

        # Query function to filter records
        def query_fn(tensor: torch.Tensor, metadata: Dict[str, Any]) -> bool:
            return metadata.get(series_metadata_field) == series_name

        matching_records = storage.query(dataset_name, query_fn)

        if not matching_records:
            logging.warning(f"No records found for series '{series_name}' in dataset '{dataset_name}'.")
            return None

        # For simplicity, use the first matching record.
        # A more robust solution might sort by a version or timestamp.
        if len(matching_records) > 1:
            logging.warning(f"Multiple records found for '{series_name}'. Using the first one (record_id: {matching_records[0]['metadata'].get('record_id')}).")
        
        record = matching_records[0]
        tensor_data = record['tensor']
        metadata = record['metadata']

        if not isinstance(tensor_data, torch.Tensor) or tensor_data.ndim != 1:
            logging.error(f"Tensor for series '{series_name}' is not a 1D torch.Tensor. Found shape: {tensor_data.shape if isinstance(tensor_data, torch.Tensor) else type(tensor_data)}")
            return None
        
        values = tensor_data.cpu().numpy() # Convert tensor to numpy array

        dates_list = None
        if date_metadata_field and date_metadata_field in metadata:
            dates_list = metadata[date_metadata_field]
            if isinstance(dates_list, list) and len(dates_list) == len(values):
                try:
                    index = pd.to_datetime(dates_list)
                    logging.info(f"Successfully parsed dates from metadata field '{date_metadata_field}'.")
                except Exception as e:
                    logging.warning(f"Could not parse dates from metadata field '{date_metadata_field}': {e}. Falling back to RangeIndex.")
                    index = pd.RangeIndex(start=0, stop=len(values), step=1)
            else:
                logging.warning(f"Date metadata field '{date_metadata_field}' found but is not a list or does not match data length. Using RangeIndex.")
                index = pd.RangeIndex(start=0, stop=len(values), step=1)
        else:
            logging.info(f"No valid date metadata field '{date_metadata_field}' found. Using RangeIndex.")
            index = pd.RangeIndex(start=0, stop=len(values), step=1)
            
        series = pd.Series(values, index=index, name=series_name)
        logging.info(f"Successfully loaded series '{series_name}' with {len(series)} data points.")
        return series

    except Exception as e:
        logging.error(f"An unexpected error occurred while loading series '{series_name}': {e}", exc_info=True)
        return None


def train_arima_and_predict(
    series: pd.Series,
    arima_order: Tuple[int, int, int] = (5, 1, 0),
    n_predictions: int = 30
) -> Optional[pd.Series]:
    """
    Trains an ARIMA model and generates future predictions.

    Args:
        series: The input time series (Pandas Series).
        arima_order: Tuple (p,d,q) for the ARIMA model.
        n_predictions: Number of future steps to predict.

    Returns:
        A Pandas Series of predictions, or None if an error occurs.
    """
    logging.info(f"Training ARIMA model with order {arima_order} for series '{series.name}' and predicting {n_predictions} steps.")
    
    if not isinstance(series, pd.Series):
        logging.error("Input 'series' must be a Pandas Series.")
        return None
    if series.empty:
        logging.error("Input 'series' is empty.")
        return None

    # Simple check for non-stationarity if d=0
    if arima_order[1] == 0:
        # A very basic check, proper stationarity tests (ADF, KPSS) are more robust
        if series.is_monotonic_increasing or series.is_monotonic_decreasing:
             logging.warning(f"ARIMA order d=0 but data for '{series.name}' appears to be monotonic. Consider differencing (increasing 'd' parameter).")

    try:
        model = ARIMA(series, order=arima_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit()
        logging.info(f"ARIMA model fitted successfully for '{series.name}'. Summary:\n{fitted_model.summary()}")

        # Generate predictions
        forecast = fitted_model.get_forecast(steps=n_predictions)
        predictions_series = forecast.predicted_mean

        # Create a future index for the predictions
        if isinstance(series.index, pd.DatetimeIndex):
            last_date = series.index[-1]
            future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_predictions, freq=series.index.freqstr or 'D')
        else: # RangeIndex
            last_val = series.index[-1]
            future_index = pd.RangeIndex(start=last_val + 1, periods=n_predictions, step=1)
        
        predictions_series.index = future_index
        predictions_series.name = f"{series.name}_predictions"

        logging.info(f"Successfully generated {n_predictions} predictions for '{series.name}'.")
        return predictions_series

    except Exception as e:
        logging.error(f"Error during ARIMA model training or prediction for '{series.name}': {e}", exc_info=True)
        return None


def store_predictions_to_tensorus(
    storage: TensorStorage,
    predictions: pd.Series,
    prediction_dataset_name: str,
    original_series_name: str,
    model_details: Dict[str, Any]
) -> Optional[str]:
    """
    Stores prediction series as a tensor in TensorStorage.

    Args:
        storage: Instance of TensorStorage.
        predictions: Pandas Series of predicted values.
        prediction_dataset_name: Dataset name to store predictions.
        original_series_name: Name of the original series that was predicted.
        model_details: Dictionary with details about the model used (e.g., ARIMA order).

    Returns:
        The record_id of the stored prediction tensor, or None if an error occurs.
    """
    logging.info(f"Storing predictions for '{original_series_name}' into dataset '{prediction_dataset_name}'.")
    try:
        if not isinstance(predictions, pd.Series):
            logging.error("Input 'predictions' must be a Pandas Series.")
            return None

        # Create dataset if it doesn't exist
        if prediction_dataset_name not in storage.list_datasets():
            logging.info(f"Prediction dataset '{prediction_dataset_name}' not found. Creating it.")
            storage.create_dataset(prediction_dataset_name)

        # Convert prediction Series to PyTorch tensor
        prediction_tensor = torch.tensor(predictions.values, dtype=torch.float32)

        # Prepare metadata
        metadata = {
            "original_series_name": original_series_name,
            "prediction_for_series": predictions.name, # e.g., "dummy_stock_close_predictions"
            "model_details": model_details,
            "prediction_start_date": str(predictions.index[0]) if isinstance(predictions.index, pd.DatetimeIndex) else int(predictions.index[0]),
            "prediction_end_date": str(predictions.index[-1]) if isinstance(predictions.index, pd.DatetimeIndex) else int(predictions.index[-1]),
            "num_predictions": len(predictions),
            "timestamp_utc": time.time(), # Current time for versioning/tracking
            "source": "time_series_predictor"
        }
        
        # If predictions have a DatetimeIndex, store the dates as metadata as well
        if isinstance(predictions.index, pd.DatetimeIndex):
            metadata["prediction_dates"] = predictions.index.strftime('%Y-%m-%d').tolist()


        record_id = storage.insert(prediction_dataset_name, prediction_tensor, metadata)
        logging.info(f"Successfully stored predictions for '{original_series_name}' with record_id '{record_id}' in dataset '{prediction_dataset_name}'.")
        return record_id

    except Exception as e:
        logging.error(f"Error storing predictions for '{original_series_name}': {e}", exc_info=True)
        return None


if __name__ == "__main__":
    logging.info("--- Starting Time Series Predictor Demo ---")

    # 1. Instantiate TensorStorage (in-memory)
    # Use a non-default storage path for demo to avoid conflicts if main app uses "tensor_data"
    demo_storage_path = "tensor_data_timeseries_demo" 
    storage_instance = TensorStorage(storage_path=demo_storage_path)
    logging.info(f"TensorStorage initialized for demo at: ./{demo_storage_path}")

    # Clean up old demo data (optional, good for repeatable tests)
    if demo_storage_path:
        import shutil
        if pd.io.common.Path(demo_storage_path).exists():
            logging.info(f"Cleaning up old demo storage directory: {demo_storage_path}")
            shutil.rmtree(demo_storage_path)
        # Re-initialize storage after cleaning path
        storage_instance = TensorStorage(storage_path=demo_storage_path)


    # 2. Create a dummy dataset and insert a sample time series
    raw_data_dataset = "stock_time_series_raw"
    series_id_name = "dummy_stock_A_close"
    
    if raw_data_dataset not in storage_instance.list_datasets():
        storage_instance.create_dataset(raw_data_dataset)
        logging.info(f"Created dataset: {raw_data_dataset}")

    num_days = 100
    # Generate slightly more interesting data for ARIMA
    base_price = 50
    trend_dummy = np.arange(num_days) * 0.2
    seasonality_dummy = 10 * np.sin(np.arange(num_days) * 2 * np.pi / 30) # Monthly seasonality
    noise_dummy = np.random.randn(num_days) * 2
    dummy_values = base_price + trend_dummy + seasonality_dummy + noise_dummy
    dummy_values = np.maximum(dummy_values, 1.0) # Ensure positive prices

    dummy_tensor = torch.tensor(dummy_values, dtype=torch.float32)
    
    sample_dates = pd.date_range(start='2023-01-01', periods=num_days).strftime('%Y-%m-%d').tolist()
    dummy_metadata = {
        "name": series_id_name, # This is what 'series_metadata_field' will look for
        "source": "dummy_generator_main",
        "currency": "USD",
        "dates": sample_dates # This is what 'date_metadata_field' will look for
    }
    
    inserted_id = storage_instance.insert(raw_data_dataset, dummy_tensor, dummy_metadata)
    if inserted_id:
        logging.info(f"Inserted dummy time series '{series_id_name}' with ID '{inserted_id}' into '{raw_data_dataset}'.")
    else:
        logging.error(f"Failed to insert dummy time series '{series_id_name}'.")
        exit() # Cannot proceed if insertion fails

    # 3. Call load_time_series_from_tensorus
    logging.info(f"\n--- Loading time series '{series_id_name}' ---")
    loaded_series = load_time_series_from_tensorus(
        storage=storage_instance,
        dataset_name=raw_data_dataset,
        series_metadata_field="name", # Matches "name" in dummy_metadata
        series_name=series_id_name,   # Matches value of "name"
        date_metadata_field="dates"   # Matches "dates" in dummy_metadata
    )

    if loaded_series is not None:
        logging.info(f"Successfully loaded series '{loaded_series.name}'. Length: {len(loaded_series)}")
        logging.info(f"Series head:\n{loaded_series.head()}")

        # 4. If data is loaded, call train_arima_and_predict
        logging.info(f"\n--- Training ARIMA and predicting for '{loaded_series.name}' ---")
        arima_params = (5, 1, 2) # (p,d,q) - choose something that might work for dummy data
        num_future_predictions = 15
        
        predictions = train_arima_and_predict(
            series=loaded_series,
            arima_order=arima_params,
            n_predictions=num_future_predictions
        )

        if predictions is not None:
            logging.info(f"Successfully generated {len(predictions)} predictions for '{loaded_series.name}'.")
            logging.info(f"Predictions head:\n{predictions.head()}")

            # 5. If predictions are made, call store_predictions_to_tensorus
            logging.info(f"\n--- Storing predictions for '{loaded_series.name}' ---")
            predictions_dataset = "stock_predictions_arima"
            model_info = {"type": "ARIMA", "order": arima_params, "parameters_estimated": True}
            
            prediction_record_id = store_predictions_to_tensorus(
                storage=storage_instance,
                predictions=predictions,
                prediction_dataset_name=predictions_dataset,
                original_series_name=loaded_series.name,
                model_details=model_info
            )

            if prediction_record_id:
                logging.info(f"Predictions stored successfully with record_id: {prediction_record_id} in dataset '{predictions_dataset}'.")

                # Verify by retrieving
                logging.info(f"\n--- Verifying stored predictions ---")
                retrieved_prediction_data = storage_instance.get_tensor_by_id(predictions_dataset, prediction_record_id)
                if retrieved_prediction_data:
                    logging.info(f"Retrieved prediction metadata: {retrieved_prediction_data['metadata']}")
                    retrieved_tensor = retrieved_prediction_data['tensor']
                    logging.info(f"Retrieved prediction tensor shape: {retrieved_tensor.shape}, first 5 values: {retrieved_tensor[:5].tolist()}")
                    # Check if data matches
                    assert torch.allclose(torch.tensor(predictions.values, dtype=torch.float32), retrieved_tensor), "Retrieved tensor data does not match original predictions!"
                    logging.info("Data consistency check passed for stored predictions.")
                else:
                    logging.error("Failed to retrieve stored predictions for verification.")
            else:
                logging.error("Failed to store predictions.")
        else:
            logging.error(f"Failed to generate ARIMA predictions for '{loaded_series.name}'.")
    else:
        logging.error(f"Failed to load series '{series_id_name}' from TensorStorage.")

    logging.info("\n--- Time Series Predictor Demo Finished ---")

    # Optional: Clean up the demo storage directory at the end
    # if demo_storage_path and pd.io.common.Path(demo_storage_path).exists():
    #     logging.info(f"Cleaning up demo storage directory: {demo_storage_path}")
    #     shutil.rmtree(demo_storage_path)
    # else:
    #     logging.info(f"Demo storage directory '{demo_storage_path}' not found or not specified, skipping cleanup.")
