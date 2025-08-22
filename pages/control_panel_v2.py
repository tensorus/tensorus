# pages/control_panel_v2.py

import streamlit as st
import time

# Import from the newly created shared utils for pages
try:
    from pages.pages_shared_utils import (
        get_agent_status,
        start_agent,
        stop_agent,
        load_css as load_shared_css
    )
except ImportError:
    st.error("Critical Error: Could not import `pages_shared_utils`. Page cannot function.")
    def get_agent_status(): st.error("`get_agent_status` unavailable."); return None
    def start_agent(agent_id): st.error(f"Start action for {agent_id} unavailable."); return False
    def stop_agent(agent_id): st.error(f"Stop action for {agent_id} unavailable."); return False
    def load_shared_css(): pass
    st.stop()


st.set_page_config(page_title="Agent Control Tower (V2)", layout="wide")
load_shared_css() # Load shared styles first. This is crucial for base styling.

# Custom CSS for Agent Cards on this page.
# These styles complement or override the .common-card style from shared_utils.
# Note: .common-card styles (background, border, padding, shadow) are inherited.
st.markdown("""
<style>
    /* Agent Card specific styling */
    .agent-card { /* This class is applied along with .common-card */
        /* Example: if agent cards needed a slightly different background or padding than common-card */
        /* background-color: #1c2849; */ /* Slightly different background if needed */
    }
    .agent-card-header { /* Header section within an agent card */
        display: flex;
        align-items: center;
        margin-bottom: 1rem; 
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #2a3f5c; /* Consistent border color */
    }
    .agent-card-header .icon { /* Icon in the header */
        font-size: 2em; 
        margin-right: 0.75rem;
        color: #9090ff; /* Specific icon color for agent cards */
    }
    .agent-card-header h3 { /* Agent name in the header */
        /* Inherits .common-card h3 styles primarily, ensuring consistency */
        color: #d0d0ff !important; /* Override if a brighter color is needed than common-card h3 */
        font-size: 1.4em !important;
        margin: 0 !important; /* Remove any default margins */
    }
    .agent-card .status-text { 
        font-weight: bold !important; /* Ensure status text is bold */
        display: inline-block; /* Allows padding and consistent look */
        padding: 0.2em 0.5em;
        border-radius: 4px;
        font-size: 0.9em;
    }
    /* Specific status colors for agent cards, using shared .status-indicator naming convention for consistency */
    .agent-card .status-running { color: #ffffff; background-color: #4CAF50;} 
    .agent-card .status-stopped { color: #e0e0e0; background-color: #525252;} 
    .agent-card .status-error   { color: #ffffff; background-color: #F44336;} 
    .agent-card .status-unknown { color: #333333; background-color: #BDBDBD;} 

    .agent-card p.description { /* Agent description text */
        font-size: 0.95em; 
        color: #c0c0e0; 
        margin-bottom: 0.75rem; 
    }
    .agent-card .metrics { /* Styling for the metrics block within an agent card */
        font-size: 0.9em;
        color: #b0b0d0;
        margin-top: 0.75rem;
        padding: 0.6rem; /* Increased padding */
        background-color: rgba(0,0,0,0.15); /* Slightly darker background for metrics section */
        border-radius: 4px;
        border-left: 3px solid #3a6fbf; /* Accent blue border */
    }
    .agent-card .metrics p { margin-bottom: 0.3rem; } /* Spacing for lines within metrics block */

    .agent-card .actions { /* Container for action buttons */
        margin-top: 1.25rem; 
        display: flex;
        gap: 0.75rem; /* Space between buttons */
        justify-content: flex-start; /* Align buttons to the start */
    }
    /* Buttons within .actions will inherit general .stButton styling from shared_css */
</style>
""", unsafe_allow_html=True)

st.title("🕹️ Agent Control Tower") 
st.caption("Manage and monitor your Tensorus intelligent agents.")

# AGENT_DETAILS_MAP provides static information (name, icon, description) for each agent type.
# This map is used to render the UI elements for agents known to the frontend.
# Metrics template helps in displaying placeholder metrics consistently.
AGENT_DETAILS_MAP = {
    "ingestion": {"name": "Ingestion Agent", "icon": "📥", "description": "Monitors file systems and ingests new data tensors into the platform.", "metrics_template": "Files Processed: {files_processed}"},
    "rl_trainer": {"name": "RL Training Agent", "icon": "🧠", "description": "Trains reinforcement learning models using available experiences and data.", "metrics_template": "Training Cycles: {episodes_trained}, Avg. Reward: {avg_reward}"},
    "automl_search": {"name": "AutoML Search Agent", "icon": "✨", "description": "Performs automated machine learning model searches and hyperparameter tuning.", "metrics_template": "Search Trials: {trials_completed}, Best Model Score: {best_score}"},
    "nql_query": {"name": "NQL Query Agent", "icon": "🗣️", "description": "Processes Natural Query Language (NQL) requests against tensor data.", "metrics_template": "Queries Handled: {queries_processed}"}
}

# Create two tabs: one for the roster of agents, another for visualizing interactions.
tab1, tab2 = st.tabs(["📊 Agent Roster", "🔗 Agent Interaction Visualizer"])

# --- Agent Roster Tab ---
with tab1:
    st.header("Live Agent Roster") # Standard header, styled by shared CSS
    
    # Button to manually refresh agent statuses from the backend.
    if st.button("🔄 Refresh Agent Statuses"):
        # Update session state cache with fresh data from API.
        # Using a page-specific cache key to avoid conflicts.
        st.session_state.agent_statuses_cache_control_panel_v2 = get_agent_status() 
        # Streamlit automatically reruns the script on button press.

    # Cache for agent statuses to avoid redundant API calls on every interaction.
    # Initializes if key is not present or if cache is None (e.g., after an error).
    cache_key = 'agent_statuses_cache_control_panel_v2'
    if cache_key not in st.session_state or st.session_state[cache_key] is None:
        st.session_state[cache_key] = get_agent_status() # Initial fetch if not in cache
    
    agents_api_data = st.session_state[cache_key] # Retrieve cached data.

    # Check if agent data was successfully fetched.
    if not agents_api_data:
        st.error("Could not fetch agent statuses. Ensure the backend API is running and `pages_shared_utils.py` is correctly configured.")
    else:
        num_columns = 2  # Define number of columns for agent card layout.
        cols = st.columns(num_columns)
        
        # Iterate through a predefined list of agent IDs expected to be available.
        # This ensures consistent ordering and display of known agents.
        agent_ids_to_display = list(AGENT_DETAILS_MAP.keys())

        for i, agent_id in enumerate(agent_ids_to_display):
            # Retrieve static details (icon, name, description) from the predefined map.
            # Provides a fallback if an agent_id from the API isn't in the map (e.g., new agent not yet in UI map).
            agent_static_details = AGENT_DETAILS_MAP.get(agent_id, {
                "name": agent_id.replace("_", " ").title() + " Agent (Unknown)", # Default name formatting
                "icon": "❓", # Default icon for unknown agent types
                "description": "Details not defined in AGENT_DETAILS_MAP.",
                "metrics_template": "Status: {status}" # Basic fallback metric
            })
            # Retrieve live runtime information (status, specific metrics) from the API data.
            agent_runtime_info = agents_api_data.get(agent_id, {}) # Default to empty dict if agent not in API response

            with cols[i % num_columns]: # Distribute agents into the defined columns.
                # Each agent is displayed in a styled card.
                # The card uses 'common-card' class from shared CSS and page-specific 'agent-card'.
                st.markdown(f'<div class="common-card agent-card" id="agent-card-{agent_id}">', unsafe_allow_html=True)
                
                # Agent card header: Icon and Name.
                st.markdown(f"""
                    <div class="agent-card-header">
                        <span class="icon">{agent_static_details['icon']}</span>
                        <h3>{agent_static_details['name']}</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Agent status: Fetched live, styled based on status string.
                # Prioritize 'status' field, fallback to 'running' boolean if 'status' is missing.
                current_status = agent_runtime_info.get('status', agent_runtime_info.get('running', 'unknown'))
                if isinstance(current_status, bool): # Convert boolean status to string representation
                    current_status = "running" if current_status else "stopped"
                status_class = f"status-{current_status.lower()}" # CSS class for styling the status text
                st.markdown(f'<p>Status: <span class="status-text {status_class}">{current_status.upper()}</span></p>', unsafe_allow_html=True)
                
                # Agent description from the static map.
                st.markdown(f"<p class='description'>{agent_static_details['description']}</p>", unsafe_allow_html=True)

                # Placeholder for agent-specific metrics.
                # Actual metrics would be populated from `agent_runtime_info` if the backend provides them.
                # Fallback to "N/A" if a metric key is not present in runtime_info.
                metrics_values = { 
                    "files_processed": agent_runtime_info.get("files_processed", "N/A"),
                    "episodes_trained": agent_runtime_info.get("episodes_trained", "N/A"),
                    "avg_reward": agent_runtime_info.get("avg_reward", "N/A"),
                    "trials_completed": agent_runtime_info.get("trials_completed", "N/A"),
                    "best_score": agent_runtime_info.get("best_score", "N/A"),
                    "queries_processed": agent_runtime_info.get("queries_processed", "N/A"),
                    "status": current_status # Fallback for generic status display in metrics
                }
                metrics_str = agent_static_details['metrics_template'].format(**metrics_values)
                st.markdown(f'<div class="metrics"><p>{metrics_str}</p></div>', unsafe_allow_html=True)

                # Action buttons container.
                st.markdown('<div class="actions">', unsafe_allow_html=True)
                actions_cols = st.columns([1,1,1]) # Columns for button layout within the actions div.
                
                with actions_cols[0]: # Details button (currently a placeholder).
                    st.button("Details", key=f"details_{agent_id}_v2", help="View detailed agent information (coming soon).", disabled=True, use_container_width=True)
                
                with actions_cols[1]: # Start/Stop button, conditional on agent status.
                    if current_status == "running":
                        if st.button("Stop", key=f"stop_{agent_id}_v2", type="secondary", use_container_width=True):
                            result = stop_agent(agent_id) # API call from shared_utils
                            st.toast(result.get("message", f"Stop request sent for {agent_id}."))
                            time.sleep(1.0) # Brief pause to allow backend to process the request.
                            st.session_state[cache_key] = get_agent_status() # Refresh status cache.
                            st.rerun() # Rerun page to reflect updated status.
                    else: # Agent is stopped, errored, or in an unknown state.
                        if st.button("Start", key=f"start_{agent_id}_v2", type="primary", use_container_width=True):
                            result = start_agent(agent_id) # API call from shared_utils
                            st.toast(result.get("message", f"Start request sent for {agent_id}."))
                            time.sleep(1.0)
                            st.session_state[cache_key] = get_agent_status() # Refresh.
                            st.rerun()
                
                with actions_cols[2]: # Configure button (currently a placeholder).
                    st.button("Configure", key=f"config_{agent_id}_v2", help="Configure agent settings (coming soon).", disabled=True, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True) # Close actions div.
                st.markdown('</div>', unsafe_allow_html=True) # Close agent-card div.

# --- Agent Interaction Visualizer Tab ---
with tab2: # Agent Interaction Visualizer Tab
    st.header("Agent Interaction Visualizer") # Standard header
    st.subheader("Conceptual Agent-Data Flow")

    # DOT language definition for the Graphviz chart.
    # This describes the nodes (agents, data stores, sources) and edges (data flow) of the system.
    graphviz_code = """
    digraph AgentDataFlow {
        // General graph attributes
        rankdir=LR; /* Layout direction: Left to Right */
        bgcolor="#0a0f2c"; /* Background color to match the app theme */
        node [shape=record, style="filled,rounded", fillcolor="#18223f", /* Default node style */
              fontname="Arial", fontsize=11, fontcolor="#e0e0e0", /* Default font attributes */
              color="#3a6fbf", penwidth=1.5]; /* Default border color and width */
        edge [fontname="Arial", fontsize=10, fontcolor="#c0c0ff", /* Default edge style */
              color="#7080ff", penwidth=1.2]; /* Default edge color and width */

        // Cluster for Data Sources
        subgraph cluster_data_sources {
            label="External Sources"; /* Cluster title */
            style="rounded";
            color="#4A5C85"; /* Cluster border color */
            bgcolor="#101828"; /* Cluster background (slightly different from main) */
            fontcolor="#D0D0FF"; /* Cluster title color */
            
            fs [label="FileSystem | (e.g., S3, Local Disk)", shape=folder, fillcolor="#3E5F8A"]; // File system node
            user_queries [label="User Queries | (via UI/API)", shape=ellipse, fillcolor="#5DADE2"]; // User queries node
        }

        // Cluster for Intelligent Agents
        subgraph cluster_agents_group {
            label="Intelligent Agents";
            style="rounded"; color="#4A5C85"; bgcolor="#101828"; fontcolor="#D0D0FF";
            
            ingestion_agent [label="{IngestionAgent | 📥 | Ingests raw data}", shape=Mrecord, fillcolor="#E74C3C"]; // Red
            nql_agent [label="{NQLAgent | 🗣️ | Processes natural language queries}", shape=Mrecord, fillcolor="#9B59B6"]; // Purple
            rl_agent [label="{RLAgent | 🧠 | Trains RL models, generates experiences}", shape=Mrecord, fillcolor="#2ECC71"]; // Green
            automl_agent [label="{AutoMLAgent | ✨ | Conducts AutoML searches}", shape=Mrecord, fillcolor="#F1C40F"]; // Yellow
        }
        
        // Cluster for Data Stores
        subgraph cluster_data_stores {
            label="Tensorus Data Stores";
            style="rounded"; color="#4A5C85"; bgcolor="#101828"; fontcolor="#D0D0FF";
            
            ingested_data_api [label="Ingested Data Store | (Primary Tensor Collection)", shape=cylinder, fillcolor="#7F8C8D", height=1.5]; // Grey
            rl_states [label="RL States | (Tensor Collection)", shape=cylinder, fillcolor="#95A5A6"]; // Lighter Grey
            rl_experiences [label="RL Experiences | (Metadata/Tensor Collection)", shape=cylinder, fillcolor="#95A5A6"];
            automl_results [label="AutoML Results | (Tensor/Metadata Collection)", shape=cylinder, fillcolor="#95A5A6"];
        }

        // Edges defining data flow between nodes
        fs -> ingestion_agent [label=" new files/streams"];
        ingestion_agent -> ingested_data_api [label=" stores tensors & metadata"];
        
        user_queries -> nql_agent [label=" NQL query"];
        nql_agent -> ingested_data_api [label=" reads data"];
        nql_agent -> rl_experiences [label=" (can also query specific stores like experiences)", style=dashed]; /* Dashed for optional/secondary path */
        
        ingested_data_api -> rl_agent [label=" reads training data/states"];
        rl_agent -> rl_states [label=" stores/updates policy states"];
        rl_agent -> rl_experiences [label=" stores new experiences"];
        rl_experiences -> rl_agent [label=" reads past experiences for training"]; // Loop back for learning
        
        ingested_data_api -> automl_agent [label=" reads datasets for AutoML"];
        automl_agent -> automl_results [label=" stores trial results & models"];
    }
    """
    try:
        # Render the Graphviz chart.
        st.graphviz_chart(graphviz_code)
        st.caption("This is a conceptual representation of data flows. Actual interactions can be more complex and configurable.")
    except Exception as e:
        # Handle potential errors if Graphviz is not installed or there's an issue with the DOT code.
        st.error(f"Could not render Graphviz chart: {e}")
    st.markdown("Please ensure Graphviz is installed and accessible in your environment's PATH (e.g., `sudo apt-get install graphviz` on Debian/Ubuntu).")

# Initialize session state key if not already present for this page.
# This helps prevent errors if the page is loaded before the cache is populated.
if 'agent_statuses_cache_control_panel_v2' not in st.session_state:
    st.session_state.agent_statuses_cache_control_panel_v2 = None
