# pages/2_Control_Panel.py (Modifications for Step 3)

import streamlit as st
import time
import json
# Use the updated API-backed functions
from .api_client import (
    list_all_agents,
    get_agent_status,
    get_agent_logs,
    start_agent,
    stop_agent,
    update_agent_config,
)

st.set_page_config(page_title="Agent Control Panel", layout="wide")

st.title("🕹️ Multi-Agent Control Panel")
st.caption("Manage and monitor Tensorus agents via API.")

# Fetch agent list from API
agent_list = list_all_agents()

if not agent_list:
    st.error("Could not fetch agent list from API. Please ensure the backend is running and reachable.")
    st.stop()

# Create a mapping from name to ID for easier selection
# Handle potential duplicate names if necessary, though IDs should be unique
agent_options = {agent['name']: agent['id'] for agent in agent_list}
# Add ID to name if names aren't unique (optional robustness)
# agent_options = {f"{agent['name']} ({agent['id']})": agent['id'] for agent in agent_list}


selected_agent_name = st.selectbox("Select Agent:", options=agent_options.keys())

if selected_agent_name:
    selected_agent_id = agent_options[selected_agent_name]
    st.divider()
    st.subheader(f"Control: {selected_agent_name} (`{selected_agent_id}`)")

    # Use session state to store fetched status and logs for the selected agent
    # This avoids refetching constantly unless a refresh is triggered
    agent_state_key = f"agent_status_{selected_agent_id}"
    agent_logs_key = f"agent_logs_{selected_agent_id}"

    # Button to force refresh status and logs
    if st.button(f"🔄 Refresh Status & Logs##{selected_agent_id}"): # Unique key per agent
        st.session_state[agent_state_key] = get_agent_status(selected_agent_id)
        st.session_state[agent_logs_key] = get_agent_logs(selected_agent_id)
        st.rerun() # Rerun to display refreshed data

    # Fetch status if not in session state or refresh button wasn't just clicked
    if agent_state_key not in st.session_state:
        st.session_state[agent_state_key] = get_agent_status(selected_agent_id)

    status_info = st.session_state[agent_state_key]

    if status_info:
        status = status_info.get('status', 'unknown')
        status_color = "green" if status in ["running", "starting"] else ("orange" if status in ["stopping"] else ("red" if status in ["error"] else "grey"))
        st.markdown(f"Current Status: :{status_color}[**{status.upper()}**]")
        last_log_ts = status_info.get('last_log_timestamp')
        if last_log_ts:
            st.caption(f"Last Log Entry (approx.): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_log_ts))}")
    else:
        st.error(f"Could not retrieve status for agent '{selected_agent_name}'.")

    # Control Buttons (Now call API functions)
    col1, col2, col3 = st.columns([1, 1, 5])
    is_running = status_info and status_info.get('status') == 'running'
    is_stopped = status_info and status_info.get('status') == 'stopped'

    with col1:
        start_disabled = not is_stopped # Disable if not stopped
        if st.button("▶️ Start", key=f"start_{selected_agent_id}", disabled=start_disabled):
            if start_agent(selected_agent_id): # API call returns success/fail
                # Trigger refresh after short delay to allow backend state change (optimistic)
                time.sleep(1.0)
                # Clear state cache and rerun
                if agent_state_key in st.session_state: del st.session_state[agent_state_key]
                if agent_logs_key in st.session_state: del st.session_state[agent_logs_key]
                st.rerun()
    with col2:
        stop_disabled = not is_running # Disable if not running
        if st.button("⏹️ Stop", key=f"stop_{selected_agent_id}", disabled=stop_disabled):
            if stop_agent(selected_agent_id): # API call returns success/fail
                time.sleep(1.0)
                if agent_state_key in st.session_state: del st.session_state[agent_state_key]
                if agent_logs_key in st.session_state: del st.session_state[agent_logs_key]
                st.rerun()

    st.divider()

    # Configuration & Logs
    tab1, tab2 = st.tabs(["Configuration", "Logs"])

    with tab1:
        if status_info and 'config' in status_info:
            current_config = status_info['config']
            st.write("Current configuration:")
            st.json(current_config)

            with st.expander("Edit configuration"):
                form = st.form(key=f"cfg_form_{selected_agent_id}")
                config_text = form.text_area(
                    "Configuration JSON",
                    value=json.dumps(current_config, indent=2),
                    height=200,
                )
                submitted = form.form_submit_button("Update")
                if submitted:
                    try:
                        new_cfg = json.loads(config_text)
                        if update_agent_config(selected_agent_id, new_cfg):
                            if agent_state_key in st.session_state:
                                del st.session_state[agent_state_key]
                            time.sleep(0.5)
                            st.rerun()
                    except json.JSONDecodeError as e:
                        form.error(f"Invalid JSON: {e}")
        else:
            st.warning("Configuration not available.")

    with tab2:
        st.write("Recent logs (fetched from API):")
        # Fetch logs if not in session state
        if agent_logs_key not in st.session_state:
            st.session_state[agent_logs_key] = get_agent_logs(selected_agent_id)

        logs = st.session_state[agent_logs_key]
        if logs is not None:
            st.code("\n".join(logs), language="log")
        else:
            st.error("Could not retrieve logs.")

else:
    st.info("Select an agent from the dropdown above.")
