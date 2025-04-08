# pages/3_💬_NQL_Chatbot.py

import streamlit as st
from ui_utils import execute_nql_query

st.set_page_config(page_title="NQL Chatbot", layout="wide")

st.title("💬 Natural Query Language (NQL) Chatbot")
st.caption("Query Tensorus datasets using natural language.")
st.info("Backend uses Regex-based NQL Agent. LLM integration is future work.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "results" in message and message["results"]:
            st.dataframe(message["results"], use_container_width=True) # Display results as dataframe
        elif "error" in message:
            st.error(message["error"])


# React to user input
if prompt := st.chat_input("Enter your query (e.g., 'get all data from my_dataset')"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response from NQL Agent API
    with st.spinner("Processing query..."):
        nql_response = execute_nql_query(prompt)

        response_content = ""
        results_df = None
        error_msg = None

        if nql_response:
            response_content = nql_response.get("message", "Error processing response.")
            if nql_response.get("success"):
                results_list = nql_response.get("results")
                if results_list:
                    # Convert results list (containing dicts with 'metadata', 'shape', etc.) to DataFrame
                    # Extract relevant fields for display
                    display_data = []
                    for res in results_list:
                        row = {
                            "record_id": res["metadata"].get("record_id"),
                            "shape": str(res.get("shape")), # Convert shape list to string
                            "dtype": res.get("dtype"),
                            **res["metadata"] # Flatten metadata into columns
                        }
                         # Remove potentially large 'tensor' data from direct display
                        row.pop('tensor', None)
                        # Avoid duplicate metadata keys if also present at top level
                        row.pop('shape', None)
                        row.pop('dtype', None)
                        row.pop('record_id', None)
                        display_data.append(row)

                    if display_data:
                         results_df = pd.DataFrame(display_data)

                # Augment message if results found
                count = nql_response.get("count")
                if count is not None:
                    response_content += f" Found {count} record(s)."

            else:
                # NQL agent indicated failure (parsing or execution)
                error_msg = response_content # Use the message as the error

        else:
            # API call itself failed (connection error, etc.)
            response_content = "Failed to get response from the NQL agent."
            error_msg = response_content

    # Display assistant response in chat message container
    message_data = {"role": "assistant", "content": response_content}
    with st.chat_message("assistant"):
        st.markdown(response_content)
        if results_df is not None:
             st.dataframe(results_df, use_container_width=True)
             message_data["results"] = results_df # Store for history display if needed (might be large)
        elif error_msg:
             st.error(error_msg)
             message_data["error"] = error_msg

    # Add assistant response to chat history
    st.session_state.messages.append(message_data)