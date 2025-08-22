import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go # Using graph_objects for more control
import time
import torch

# Assuming tensorus is in PYTHONPATH or installed
from tensorus.tensor_storage import TensorStorage
from tensorus.financial_data_generator import generate_financial_data
from tensorus.time_series_predictor import load_time_series_from_tensorus, train_arima_and_predict, store_predictions_to_tensorus

# Attempt to load shared CSS
try:
    from pages.pages_shared_utils import load_css
    LOAD_CSS_AVAILABLE = True
except ImportError:
    LOAD_CSS_AVAILABLE = False
    def load_css(): # Dummy function
        st.markdown("<style>/* No shared CSS found */</style>", unsafe_allow_html=True)
        # st.warning("Could not load shared CSS from pages.pages_shared_utils.") # Keep UI cleaner

# Page Configuration
st.set_page_config(page_title="Financial Forecast Demo", layout="wide")
if LOAD_CSS_AVAILABLE:
    load_css()

# --- Constants ---
RAW_DATASET_NAME = "financial_raw_data"
PREDICTION_DATASET_NAME = "financial_predictions"
TIME_SERIES_NAME = "synthetic_stock_close"
# Use a demo-specific storage path to avoid conflicts with main app
TENSOR_STORAGE_PATH = "tensor_data_financial_demo"


def get_tensor_storage_instance():
    """Initializes and returns a TensorStorage instance for the demo."""
    return TensorStorage(storage_path=TENSOR_STORAGE_PATH)

def show_financial_demo_page():
    """
    Main function to display the Streamlit page for the financial demo.
    """
    st.title("📈 Financial Time Series Forecasting Demo")
    st.markdown("""
    This demo showcases time series forecasting using an ARIMA model on synthetically generated 
    financial data. Data is generated, stored in Tensorus, loaded for model training, 
    and predictions are then stored back and visualized.
    """)

    # Initialize session state variables
    if 'raw_financial_df' not in st.session_state:
        st.session_state.raw_financial_df = None
    if 'loaded_historical_series' not in st.session_state:
        st.session_state.loaded_historical_series = None
    if 'predictions_series' not in st.session_state:
        st.session_state.predictions_series = None
    if 'arima_p' not in st.session_state: # For number_input persistence
        st.session_state.arima_p = 5
    if 'arima_d' not in st.session_state:
        st.session_state.arima_d = 1
    if 'arima_q' not in st.session_state:
        st.session_state.arima_q = 0
    if 'n_predictions' not in st.session_state:
        st.session_state.n_predictions = 30


    # --- Section 1: Data Generation & Ingestion ---
    st.header("1. Data Generation & Ingestion")
    if st.button("Generate & Ingest Sample Financial Data", key="generate_data_button"):
        with st.spinner("Generating and ingesting data..."):
            try:
                storage = get_tensor_storage_instance()
                
                # Generate sample financial data
                df = generate_financial_data(days=365, initial_price=150, trend_slope=0.1, seasonality_amplitude=20)
                st.session_state.raw_financial_df = df # Store for immediate plotting

                # Ensure dataset exists
                if RAW_DATASET_NAME not in storage.list_datasets():
                    storage.create_dataset(RAW_DATASET_NAME)
                    st.info(f"Dataset '{RAW_DATASET_NAME}' created.")
                else:
                    st.info(f"Dataset '{RAW_DATASET_NAME}' already exists. Using existing.")

                # Prepare data for TensorStorage
                series_to_store = torch.tensor(df['Close'].values, dtype=torch.float32)
                dates_for_metadata = df['Date'].dt.strftime('%Y-%m-%d').tolist()
                metadata = {
                    "name": TIME_SERIES_NAME, 
                    "dates": dates_for_metadata,
                    "source": "financial_demo_generator",
                    "description": f"Daily closing prices for {TIME_SERIES_NAME}"
                }

                record_id = storage.insert(RAW_DATASET_NAME, series_to_store, metadata)
                if record_id:
                    st.success(f"Data for '{TIME_SERIES_NAME}' ingested into '{RAW_DATASET_NAME}' with Record ID: {record_id}")
                    # Also load it into loaded_historical_series for consistency in the workflow
                    st.session_state.loaded_historical_series = pd.Series(df['Close'].values, index=pd.to_datetime(df['Date']), name=TIME_SERIES_NAME)
                    st.session_state.predictions_series = None # Clear previous predictions
                else:
                    st.error("Failed to ingest data into Tensorus.")
            except Exception as e:
                st.error(f"Error during data generation/ingestion: {e}")

    # --- Section 2: Historical Data Visualization ---
    st.header("2. Historical Data Visualization")
    if st.button("Load and View Historical Data from Tensorus", key="load_historical_button"):
        with st.spinner(f"Loading '{TIME_SERIES_NAME}' from Tensorus..."):
            try:
                storage = get_tensor_storage_instance()
                loaded_series = load_time_series_from_tensorus(
                    storage, 
                    RAW_DATASET_NAME, 
                    series_metadata_field="name", 
                    series_name=TIME_SERIES_NAME, 
                    date_metadata_field="dates"
                )
                if loaded_series is not None:
                    st.session_state.loaded_historical_series = loaded_series
                    st.session_state.predictions_series = None # Clear previous predictions
                    st.success(f"Successfully loaded '{TIME_SERIES_NAME}' from Tensorus.")
                else:
                    st.warning(f"Could not find or load '{TIME_SERIES_NAME}' from Tensorus dataset '{RAW_DATASET_NAME}'. Generate data first if it's not there.")
            except Exception as e:
                st.error(f"Error loading historical data: {e}")

    if st.session_state.loaded_historical_series is not None:
        st.subheader(f"Historical Data: {st.session_state.loaded_historical_series.name}")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=st.session_state.loaded_historical_series.index, 
            y=st.session_state.loaded_historical_series.values, 
            mode='lines', 
            name='Historical Close'
        ))
        fig_hist.update_layout(title="Historical Stock Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Generate or load data to view historical prices.")


    # --- Section 3: ARIMA Model Prediction ---
    st.header("3. ARIMA Model Prediction")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.number_input("ARIMA Order (p)", min_value=0, max_value=10, key="arima_p", help="Autoregressive order.")
    with col2:
        st.number_input("ARIMA Order (d)", min_value=0, max_value=3, key="arima_d", help="Differencing order.")
    with col3:
        st.number_input("ARIMA Order (q)", min_value=0, max_value=10, key="arima_q", help="Moving average order.")
    with col4:
        st.number_input("Number of Future Predictions", min_value=1, max_value=180, key="n_predictions", help="Number of future days to predict.")

    if st.button("Run ARIMA Prediction", key="run_arima_button"):
        if st.session_state.loaded_historical_series is None:
            st.error("No historical data loaded. Please generate or load data first (Sections 1 or 2).")
        else:
            with st.spinner("Running ARIMA prediction... This might take a moment."):
                try:
                    p, d, q = st.session_state.arima_p, st.session_state.arima_d, st.session_state.arima_q
                    n_preds = st.session_state.n_predictions
                    
                    predictions = train_arima_and_predict(
                        series=st.session_state.loaded_historical_series,
                        arima_order=(p, d, q),
                        n_predictions=n_preds
                    )
                    
                    if predictions is not None:
                        st.session_state.predictions_series = predictions
                        st.success(f"ARIMA prediction complete for {n_preds} steps.")
                        
                        # Store predictions
                        storage = get_tensor_storage_instance()
                        model_details_dict = {"type": "ARIMA", "order": (p,d,q), "parameters_estimated": True} # Example details
                        
                        pred_record_id = store_predictions_to_tensorus(
                            storage, 
                            predictions, 
                            PREDICTION_DATASET_NAME, 
                            original_series_name=st.session_state.loaded_historical_series.name, 
                            model_details=model_details_dict
                        )
                        if pred_record_id:
                           st.info(f"Predictions stored in Tensorus dataset '{PREDICTION_DATASET_NAME}' with Record ID: {pred_record_id}")
                        else:
                           st.warning("Failed to store predictions in Tensorus.")
                    else:
                        st.error("Failed to generate ARIMA predictions. Check model parameters, data, or logs.")
                except Exception as e:
                    st.error(f"Error during ARIMA prediction: {e}")

    # --- Section 4: Prediction Results & Visualization ---
    st.header("4. Prediction Results")
    if st.session_state.predictions_series is not None and st.session_state.loaded_historical_series is not None:
        st.subheader("Historical Data and Predictions")
        
        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(
            x=st.session_state.loaded_historical_series.index, 
            y=st.session_state.loaded_historical_series.values, 
            mode='lines', 
            name='Historical Close',
            line=dict(color='blue')
        ))
        fig_combined.add_trace(go.Scatter(
            x=st.session_state.predictions_series.index, 
            y=st.session_state.predictions_series.values, 
            mode='lines', 
            name='Predicted Close',
            line=dict(color='red', dash='dash')
        ))
        fig_combined.update_layout(
            title=f"{st.session_state.loaded_historical_series.name}: Historical vs. Predicted", 
            xaxis_title="Date", 
            yaxis_title="Price"
        )
        st.plotly_chart(fig_combined, use_container_width=True)

        st.subheader("Predicted Values (Next {} Days)".format(len(st.session_state.predictions_series)))
        # Display predictions as a DataFrame, ensure index is named 'Date' if it's a DatetimeIndex
        pred_df_display = st.session_state.predictions_series.to_frame()
        if isinstance(pred_df_display.index, pd.DatetimeIndex):
            pred_df_display.index.name = "Date"
        st.dataframe(pred_df_display)

    elif st.session_state.loaded_historical_series is not None:
        st.info("Run ARIMA Prediction in Section 3 to see forecasted results.")
    else:
        st.info("Generate or load data and run predictions to see results here.")

if __name__ == "__main__":
    show_financial_demo_page()
