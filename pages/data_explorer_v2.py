# pages/data_explorer_v2.py

import streamlit as st
import pandas as pd
import plotly.express as px
import torch

try:
    from pages.pages_shared_utils import (
        load_css as load_shared_css,
        get_datasets,
        get_dataset_preview
    )
except ImportError:
    st.error("Critical Error: Could not import `pages_shared_utils`. Page cannot function.")
    def load_shared_css(): pass
    def get_datasets(): st.error("`get_datasets` unavailable."); return []
    def get_dataset_preview(dataset_name, limit=5): st.error("`get_dataset_preview` unavailable."); return None
    st.stop()

# Configure page settings
st.set_page_config(page_title="Tensor Explorer V2", layout="wide")
load_shared_css() # Apply shared Nexus theme styles

st.title("🔍 Tensor Explorer (V2)")
st.caption("Browse, filter, and visualize Tensorus datasets with agent context.")

# --- Dataset Selection ---
# Fetch the list of available datasets from the backend using shared utility.
available_datasets = get_datasets() 
if not available_datasets:
    st.warning("No datasets found. Ensure the backend API is running and accessible.")
    st.stop() # Halt execution if no datasets are available for exploration.

# Dropdown for user to select a dataset.
selected_dataset_name = st.selectbox("Select Dataset:", available_datasets)

# --- Data Fetching & Initial Processing ---
# Proceed only if a dataset is selected.
if selected_dataset_name:
    st.subheader(f"Exploring: {selected_dataset_name}") # Styled by shared CSS

    MAX_RECORDS_DISPLAY = 100 # Define max records to fetch for the initial preview.
    # Fetch dataset preview information (includes sample records) using shared utility.
    dataset_info = get_dataset_preview(selected_dataset_name, limit=MAX_RECORDS_DISPLAY)

    # Validate fetched data.
    if dataset_info is None or "preview" not in dataset_info:
        st.error(f"Failed to fetch data preview for '{selected_dataset_name}'. The backend might be down or the dataset is invalid.")
        st.stop()
    
    records_preview = dataset_info.get("preview", []) # List of records (tensor data + metadata).
    total_records_in_dataset = dataset_info.get("record_count", len(records_preview)) # Total records in the dataset.

    if not records_preview:
        st.info(f"Dataset '{selected_dataset_name}' is empty or no preview data available.")
        st.stop()

    st.info(f"Displaying {len(records_preview)} of {total_records_in_dataset} records from '{selected_dataset_name}'.")

    # Prepare data for DataFrame display: extract and flatten metadata.
    # This makes metadata fields directly accessible as columns in the DataFrame for filtering and display.
    processed_records_for_df = []
    for record in records_preview:
        meta = record.get('metadata', {}).copy() # Make a copy to avoid modifying original record.
        meta['tensor_id'] = record.get('id', 'N/A') # Use 'id' from record as 'tensor_id'.
        meta['shape'] = str(record.get('shape', 'N/A')) # Store shape as string for display.
        meta['dtype'] = record.get('dtype', 'N/A')
        # Ensure 'created_by' field exists, defaulting to 'Unknown' if not present. This is for filtering.
        meta['created_by'] = meta.get('created_by', 'Unknown')
        processed_records_for_df.append(meta)

    df_display_initial = pd.DataFrame(processed_records_for_df) # DataFrame for filtering and display.

    # --- Sidebar Filtering UI ---
    st.sidebar.header("Filter Options") # Styled by shared CSS

    # Filter by Source Agent ('created_by' field).
    st.sidebar.subheader("Filter by Source Agent") # Styled by shared CSS
    # Predefined agent sources, augmented with any unique sources found in the data for comprehensive filtering.
    agent_sources_default = ["IngestionAgent", "RLAgent", "AutoMLAgent", "Unknown"] 
    if 'created_by' in df_display_initial.columns:
        available_agents_in_data = df_display_initial['created_by'].unique().tolist()
        filter_options_agents = sorted(list(set(agent_sources_default + available_agents_in_data)))
    else:
        filter_options_agents = agent_sources_default # Fallback if 'created_by' column is missing.

    selected_agent_filters = st.sidebar.multiselect(
        "Show data created by:",
        options=filter_options_agents,
        default=[] # No agents selected by default; shows all data initially.
    )

    # Filter by Other Metadata Fields.
    st.sidebar.subheader("Filter by Other Metadata Fields") # Styled by shared CSS
    # Allow filtering on any metadata column except those already specifically handled or less useful for direct filtering.
    potential_filter_cols = [col for col in df_display_initial.columns if col not in ['tensor_id', 'created_by', 'shape', 'dtype']]
    
    filter_cols_metadata = st.sidebar.multiselect(
        "Select metadata fields to filter:",
        options=potential_filter_cols
    )

    # Apply filters to the DataFrame. Start with a copy of the initial DataFrame.
    filtered_df_final = df_display_initial.copy()

    # Apply agent source filter if any agents are selected.
    if selected_agent_filters:
        if 'created_by' in filtered_df_final.columns:
            filtered_df_final = filtered_df_final[filtered_df_final['created_by'].isin(selected_agent_filters)]
        else: 
            # This case should ideally not be reached if 'created_by' is always added or defaulted.
            st.sidebar.warning("'created_by' field not found for agent filtering.")
            filtered_df_final = filtered_df_final.iloc[0:0] # Show no results if filter is active but field is missing.
            
    # Apply other metadata filters based on user selections.
    for col in filter_cols_metadata:
        unique_values = filtered_df_final[col].dropna().unique().tolist()
        # Numeric filter: use a range slider if there are multiple unique numeric values.
        if pd.api.types.is_numeric_dtype(filtered_df_final[col]) and filtered_df_final[col].nunique() > 1:
            min_val, max_val = float(filtered_df_final[col].min()), float(filtered_df_final[col].max())
            selected_range = st.sidebar.slider(f"Filter {col}:", min_val, max_val, (min_val, max_val), key=f"slider_{col}")
            filtered_df_final = filtered_df_final[filtered_df_final[col].between(selected_range[0], selected_range[1])]
        # Categorical filter (with limited unique values for dropdown): use multiselect.
        elif len(unique_values) > 0 and len(unique_values) <= 25: # Threshold for using multiselect.
             default_selection = [] # Default to no specific selection (i.e., don't filter unless user picks).
             selected_values = st.sidebar.multiselect(f"Filter {col}:", options=unique_values, default=default_selection, key=f"multi_{col}")
             if selected_values: # Only apply filter if user has made selections.
                filtered_df_final = filtered_df_final[filtered_df_final[col].isin(selected_values)]
        # Text search for other columns or columns with too many unique values.
        else:
            search_term = st.sidebar.text_input(f"Search in {col} (contains):", key=f"text_{col}").lower()
            if search_term:
                filtered_df_final = filtered_df_final[filtered_df_final[col].astype(str).str.lower().str.contains(search_term, na=False)]

    # --- Display Filtered Data ---
    st.divider() # Visual separator.
    st.subheader("Filtered Data View") # Styled by shared CSS
    st.write(f"{len(filtered_df_final)} records matching filters.") # Display count of filtered records.
    
    # Define preferred column order for the displayed DataFrame.
    cols_to_display_order = ['tensor_id', 'created_by', 'shape', 'dtype']
    remaining_cols = [col for col in df_display_initial.columns if col not in cols_to_display_order]
    # Ensure only existing columns are included in the final display order.
    final_display_columns = [col for col in cols_to_display_order if col in filtered_df_final.columns] + \
                            [col for col in remaining_cols if col in filtered_df_final.columns]
    
    st.dataframe(filtered_df_final[final_display_columns], use_container_width=True, hide_index=True)

    # --- Tensor Preview & Visualization ---
    st.divider()
    st.subheader("Tensor Preview") # Styled by shared CSS

    if not filtered_df_final.empty:
        available_tensor_ids = filtered_df_final['tensor_id'].tolist()
        # Dropdown to select a tensor ID from the filtered results for preview.
        selected_tensor_id = st.selectbox("Select Tensor ID to Preview:", available_tensor_ids, key="tensor_preview_select")

        if selected_tensor_id:
            # Retrieve the full record (including raw tensor data list) from the original preview list.
            selected_full_record = next((r for r in records_preview if r.get('id') == selected_tensor_id), None)

            if selected_full_record:
                st.write("**Full Metadata:**")
                st.json(selected_full_record.get('metadata', {})) # Display all metadata for the selected tensor.

                shape = selected_full_record.get("shape")
                dtype_str = selected_full_record.get("dtype")
                data_list = selected_full_record.get("data") # Raw list representation of tensor data.

                st.write(f"**Tensor Info:** Shape=`{shape}`, Dtype=`{dtype_str}`")
                source_agent = selected_full_record.get('metadata', {}).get('created_by', 'Unknown')
                st.write(f"**Source Agent:** `{source_agent}`") # Display the 'created_by' agent.

                try:
                    if shape and dtype_str and data_list is not None:
                        # Reconstruct the tensor from its list representation and metadata.
                        torch_dtype = getattr(torch, dtype_str, None) # Get torch.dtype from string.
                        if torch_dtype is None:
                            st.error(f"Unsupported dtype: {dtype_str}. Cannot reconstruct tensor.")
                        else:
                            tensor = torch.tensor(data_list, dtype=torch_dtype)
                            st.write("**Tensor Data (first 10 elements flattened):**")
                            st.code(f"{tensor.flatten()[:10].cpu().numpy()}...") # Display a snippet of tensor data.

                            # --- Simple Visualizations based on tensor dimensions ---
                            if tensor.ndim == 1 and tensor.numel() > 1: # 1D tensor: line chart.
                                st.line_chart(tensor.cpu().numpy())
                            elif tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1 : # 2D tensor: heatmap.
                                try:
                                    fig = px.imshow(tensor.cpu().numpy(), title="Tensor Heatmap", aspect="auto")
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as plot_err:
                                    st.warning(f"Could not generate heatmap: {plot_err}")
                            elif tensor.ndim == 3 and tensor.shape[0] in [1, 3]: # 3D tensor (potential image): try to display as image.
                                try:
                                    display_tensor = tensor.cpu()
                                    if display_tensor.shape[0] == 1: # Grayscale image (C, H, W) -> (H, W)
                                        display_tensor = display_tensor.squeeze(0)
                                    elif display_tensor.shape[0] == 3: # RGB image (C, H, W) -> (H, W, C)
                                        display_tensor = display_tensor.permute(1, 2, 0)
                                    
                                    # Normalize for display if not in typical image range [0,1] or [0,255].
                                    if display_tensor.max() > 1.0 or display_tensor.min() < 0.0: # Basic check
                                         display_tensor = (display_tensor - display_tensor.min()) / (display_tensor.max() - display_tensor.min() + 1e-6) # Normalize to [0,1]
                                    
                                    st.image(display_tensor.numpy(), caption="Tensor as Image (Attempted)", use_column_width=True)
                                except Exception as img_err:
                                    st.warning(f"Could not display tensor as image: {img_err}")
                            else:
                                st.info("No specific visualization available for this tensor's shape/dimension.")
                    else:
                        st.warning("Tensor data, shape, or dtype missing in the selected record.")
                except Exception as tensor_err:
                    st.error(f"Error processing tensor data for preview: {tensor_err}")
            else:
                st.warning("Selected tensor ID details not found in the fetched preview data.")
        else:
            st.info("Select a Tensor ID from the filtered table above to preview its details.")
    else:
        st.info("No records match the current filters to allow preview.")
else:
    st.info("Select a dataset to start exploring.")

