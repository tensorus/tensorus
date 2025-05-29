# pages/4_explorador_Data_Explorer.py

import streamlit as st
import pandas as pd
import plotly.express as px
from .ui_utils import list_datasets, fetch_dataset_data # MODIFIED
import torch # Needed if we want to recreate tensors for inspection/plotting

st.set_page_config(page_title="Data Explorer", layout="wide")

st.title("🔍 Interactive Data Explorer")
st.caption("Browse, filter, and visualize Tensorus datasets.")

# --- Dataset Selection ---
datasets = list_datasets()
if not datasets:
    st.warning("No datasets found or API connection failed. Cannot explore data.")
    st.stop() # Stop execution if no datasets

selected_dataset = st.selectbox("Select Dataset:", datasets)

# --- Data Fetching & Filtering ---
if selected_dataset:
    st.subheader(f"Exploring: {selected_dataset}")

    # Fetch data (limited records for UI)
    # TODO: Implement server-side sampling/pagination via API for large datasets
    MAX_RECORDS_DISPLAY = 100
    records = fetch_dataset_data(selected_dataset, max_records=MAX_RECORDS_DISPLAY)

    if records is None:
        st.error("Failed to fetch data for the selected dataset.")
        st.stop()
    elif not records:
        st.info("Selected dataset is empty.")
        st.stop()

    st.info(f"Displaying first {len(records)} records out of potentially more.")

    # Create DataFrame from metadata for filtering/display
    metadata_list = [r['metadata'] for r in records]
    df_meta = pd.DataFrame(metadata_list)

    # --- Metadata Filtering UI ---
    st.sidebar.header("Filter by Metadata")
    filter_cols = st.sidebar.multiselect("Select metadata columns to filter:", options=df_meta.columns.tolist())

    filtered_df = df_meta.copy()
    for col in filter_cols:
        unique_values = filtered_df[col].dropna().unique().tolist()
        if pd.api.types.is_numeric_dtype(filtered_df[col]):
            # Numeric filter (slider)
            min_val, max_val = float(filtered_df[col].min()), float(filtered_df[col].max())
            if min_val < max_val:
                selected_range = st.sidebar.slider(f"Filter {col}:", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[filtered_df[col].between(selected_range[0], selected_range[1])]
            else:
                st.sidebar.caption(f"{col}: Single numeric value ({min_val}), no range filter.")

        elif len(unique_values) > 0 and len(unique_values) <= 20: # Limit dropdown options
            # Categorical filter (multiselect)
             selected_values = st.sidebar.multiselect(f"Filter {col}:", options=unique_values, default=unique_values)
             if selected_values: # Only filter if some values are selected
                  filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
             else: # If user deselects everything, show nothing
                  filtered_df = filtered_df[filtered_df[col].isnull()] # Hack to get empty DF matching columns

        else:
            st.sidebar.text_input(f"Filter {col} (Text contains):", key=f"text_{col}")
            search_term = st.session_state.get(f"text_{col}", "").lower()
            if search_term:
                # Ensure column is string type before using .str.contains
                filtered_df = filtered_df[filtered_df[col].astype(str).str.lower().str.contains(search_term, na=False)]


    st.divider()
    st.subheader("Filtered Data View")
    st.write(f"{len(filtered_df)} records matching filters.")
    st.dataframe(filtered_df, use_container_width=True)

    # --- Tensor Preview & Visualization ---
    st.divider()
    st.subheader("Tensor Preview")

    if not filtered_df.empty:
        # Allow selecting a record ID from the filtered results
        record_ids = filtered_df['record_id'].tolist()
        selected_record_id = st.selectbox("Select Record ID to Preview Tensor:", record_ids)

        if selected_record_id:
            # Find the full record data corresponding to the selected ID
            selected_record = next((r for r in records if r['metadata'].get('record_id') == selected_record_id), None)

            if selected_record:
                st.write("Metadata:")
                st.json(selected_record['metadata'])

                shape = selected_record.get("shape")
                dtype = selected_record.get("dtype")
                data_list = selected_record.get("data")

                st.write(f"Tensor Info: Shape={shape}, Dtype={dtype}")

                try:
                     # Recreate tensor for potential plotting/display
                     # Be careful with large tensors in Streamlit UI!
                     # We might only want to show info or small slices.
                     if shape and dtype and data_list is not None:
                          tensor = torch.tensor(data_list, dtype=getattr(torch, dtype, torch.float32)) # Use getattr for dtype
                          st.write("Tensor Data (first few elements):")
                          st.code(f"{tensor.flatten()[:10].numpy()}...") # Show flattened start

                          # --- Simple Visualizations ---
                          if tensor.ndim == 1 and tensor.numel() > 1:
                               st.line_chart(tensor.numpy())
                          elif tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 1 :
                               # Simple heatmap using plotly (requires plotly)
                               try:
                                   fig = px.imshow(tensor.numpy(), title="Tensor Heatmap", aspect="auto")
                                   st.plotly_chart(fig, use_container_width=True)
                               except Exception as plot_err:
                                   st.warning(f"Could not generate heatmap: {plot_err}")
                          elif tensor.ndim == 3 and tensor.shape[0] in [1, 3]: # Basic image check (C, H, W) or (1, H, W)
                               try:
                                   # Permute if needed (e.g., C, H, W -> H, W, C for display)
                                   if tensor.shape[0] in [1, 3]:
                                       display_tensor = tensor.permute(1, 2, 0).squeeze() # H, W, C or H, W
                                       # Clamp/normalize data to display range [0, 1] or [0, 255] - basic attempt
                                       display_tensor = (display_tensor - display_tensor.min()) / (display_tensor.max() - display_tensor.min() + 1e-6)
                                       st.image(display_tensor.numpy(), caption="Tensor as Image (Attempted)", use_column_width=True)
                               except ImportError:
                                    st.warning("Pillow needed for image display (`pip install Pillow`)")
                               except Exception as img_err:
                                    st.warning(f"Could not display tensor as image: {img_err}")
                          else:
                               st.info("No specific visualization available for this tensor shape/dimension.")

                     else:
                          st.warning("Tensor data, shape, or dtype missing in the record.")

                except Exception as tensor_err:
                    st.error(f"Error processing tensor data for preview: {tensor_err}")
            else:
                 st.warning("Selected record details not found (this shouldn't happen).")
        else:
            st.info("Select a record ID above to preview its tensor.")
    else:
        st.info("No records match the current filters.")
