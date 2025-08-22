# pages/1_Dashboard.py (Modifications for Step 3)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
# Updated imports to use API-backed functions
from .ui_utils import get_dashboard_metrics, list_all_agents, get_agent_status # MODIFIED

st.set_page_config(page_title="Tensorus Dashboard", layout="wide")

st.title("📊 Operations Dashboard")
st.caption("Overview of Tensorus datasets and agent activity from API.")

# --- Fetch Data ---
# Use st.cache_data for API calls that don't need constant updates
# or manage refresh manually. For simplicity, call directly on rerun/button click.
metrics_data = None
agent_list = None # Fetch full agent list for detailed status display

# Button to force refresh
if st.button("🔄 Refresh Dashboard Data"):
    # Clear previous cache if any or just proceed to refetch
    metrics_data = get_dashboard_metrics()
    agent_list = list_all_agents()
    st.session_state['dashboard_metrics'] = metrics_data # Store in session state
    st.session_state['dashboard_agents'] = agent_list
    st.rerun() # Rerun the script to reflect fetched data
else:
    # Try to load from session state or fetch if not present
    if 'dashboard_metrics' not in st.session_state:
         st.session_state['dashboard_metrics'] = get_dashboard_metrics()
    if 'dashboard_agents' not in st.session_state:
         st.session_state['dashboard_agents'] = list_all_agents()

    metrics_data = st.session_state['dashboard_metrics']
    agent_list = st.session_state['dashboard_agents']


# --- Display Metrics ---
st.subheader("System Metrics")
if metrics_data:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Datasets", metrics_data.get('dataset_count', 'N/A'))
    col2.metric("Total Records (Est.)", f"{metrics_data.get('total_records_est', 0):,}")
    # Agent status summary from metrics
    agent_summary = metrics_data.get('agent_status_summary', {})
    running_agents = agent_summary.get('running', 0) + agent_summary.get('starting', 0)
    col3.metric("Running Agents", running_agents)

    st.divider()

    # --- Performance Metrics Row ---
    st.subheader("Performance Indicators (Simulated)")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    pcol1.metric("Ingestion Rate (rec/s)", f"{metrics_data.get('data_ingestion_rate', 0.0):.1f}")
    pcol2.metric("Avg Query Latency (ms)", f"{metrics_data.get('avg_query_latency_ms', 0.0):.1f}")
    pcol3.metric("Latest RL Reward", f"{metrics_data.get('rl_latest_reward', 'N/A')}")
    pcol4.metric("Best AutoML Score", f"{metrics_data.get('automl_best_score', 'N/A')}")

else:
    st.warning("Could not fetch dashboard metrics from the API.")

st.divider()

# --- Agent Status Details ---
st.subheader("Agent Status")
if agent_list:
    num_agents = len(agent_list)
    cols = st.columns(max(1, num_agents)) # Create columns for agents

    for i, agent_info in enumerate(agent_list):
        agent_id = agent_info.get('id')
        with cols[i % len(cols)]: # Distribute agents into columns
            with st.container(border=True):
                st.markdown(f"**{agent_info.get('name', 'Unknown Agent')}** (`{agent_id}`)")
                # Fetch detailed status for more info if needed, or use basic status from list
                # status_details = get_agent_status(agent_id) # Can make page slower
                status = agent_info.get('status', 'unknown')
                status_color = "green" if status in ["running", "starting"] else ("orange" if status in ["stopping"] else ("red" if status in ["error"] else "grey"))
                st.markdown(f"Status: :{status_color}[**{status.upper()}**]")

                # Display config from the list info
                with st.expander("Config"):
                    st.json(agent_info.get('config', {}), expanded=False)
else:
    st.warning("Could not fetch agent list from the API.")

st.divider()

# --- Performance Monitoring Chart (Using simulated data from metrics for now) ---
st.subheader("Performance Monitoring (Placeholder Graph)")
if metrics_data:
    # Create some fake historical data for plotting based on current metrics
    history_len = 20
    # Use session state to persist some history for smoother simulation
    if 'sim_history' not in st.session_state:
        st.session_state['sim_history'] = pd.DataFrame({
            'Ingestion Rate': np.random.rand(history_len) * metrics_data.get('data_ingestion_rate', 10),
            'Query Latency': np.random.rand(history_len) * metrics_data.get('avg_query_latency_ms', 100),
            'RL Reward': np.random.randn(history_len) * 5 + (metrics_data.get('rl_latest_reward', 0) or 0)
        })

    # Update history with latest point
    latest_data = pd.DataFrame({
         'Ingestion Rate': [metrics_data.get('data_ingestion_rate', 0.0)],
         'Query Latency': [metrics_data.get('avg_query_latency_ms', 0.0)],
         'RL Reward': [metrics_data.get('rl_latest_reward', 0) or 0] # Handle None
    })
    st.session_state['sim_history'] = pd.concat([st.session_state['sim_history'].iloc[1:], latest_data], ignore_index=True)


    # Use Plotly for better interactivity
    try:
        fig = px.line(st.session_state['sim_history'], title="Simulated Performance Metrics Over Time")
        fig.update_layout(legend_title_text='Metrics')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display performance chart: {e}")

else:
     st.info("Performance metrics unavailable.")

st.caption(f"Dashboard data timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics_data.get('timestamp', time.time())) if metrics_data else time.time())}")
