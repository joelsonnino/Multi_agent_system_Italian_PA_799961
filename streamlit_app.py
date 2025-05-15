# --- START OF FILE streamlit_app.py ---

"""
Streamlit Interface for Multi-Agent System using Ollama
"""
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from orchestrator_agent import orchestrate 
from langchain_experimental.tools.python.tool import PythonAstREPLTool as PythonREPLTool
from langchain.memory import ConversationBufferWindowMemory
import logging
import traceback
import html 
from io import BytesIO 


# --- Page Configuration ---
st.set_page_config(
    page_title="NOIPA Portal Data Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# This CSS overrides the default primary button style to render them in blue.
st.markdown("""
<style>
    /* Selects primary buttons by targeting buttons without "secondary" classes.
       Streamlit applies default styles (using primaryColor) to primary buttons
       and adds specific classes for secondary buttons.
       Therefore, we target buttons that do NOT have "secondary" in their class. */
    div[data-testid="stButton"] > button:not([class*="secondary"]) {
        background-color: #007bff; /* Standard blue */
        color: white;              /* White text for contrast */
        border: 1px solid #007bff; /* Matching border */
    }
    div[data-testid="stButton"] > button:not([class*="secondary"]):hover {
        background-color: #0056b3; /* Darker blue on hover */
        color: white;
        border: 1px solid #0056b3;
    }
    div[data-testid="stButton"] > button:not([class*="secondary"]):active {
        background-color: #004085; /* Even darker blue on active/click */
        color: white;
        border: 1px solid #004085;
    }
    /*
    Optional: Style for disabled primary buttons, if you want them
    to follow the blue theme
    div[data-testid="stButton"] > button:not([class*="secondary"]):disabled {
        background-color: #a7c7e7; /* Light desaturated blue */
        color: #e0e0e0;
        border: 1px solid #a7c7e7;
    }
    */
</style>
""", unsafe_allow_html=True)



# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize Python REPL Tool ---
python_repl_globals = {"pd": pd, "np": np, "os": os, "plt": plt}
try:
    import seaborn as sns
    python_repl_globals['sns'] = sns
except ImportError:
    logger.info("Seaborn not installed, not adding to REPL globals by default.")
    pass

python_repl = PythonREPLTool(locals=python_repl_globals)


# --- Helper Functions ---
@st.cache_data
def get_dataset_filenames(folder="datasets"):
    """Lists CSV files in the specified folder."""
    try:
        if not os.path.isdir(folder):
             logger.error(f"Datasets directory '{folder}' not found. Please create it.")
             st.error(f"Critical: Datasets directory '{folder}' not found. Please create it and add CSV files.")
             return []
        files = [f for f in os.listdir(folder) if f.endswith('.csv') and os.path.isfile(os.path.join(folder, f))]
        if not files:
            logger.warning(f"No CSV files found in '{folder}'.")
        return sorted(files)
    except Exception as e:
        logger.error(f"An error occurred while listing datasets: {e}", exc_info=True)
        st.error(f"Error listing datasets: {e}")
        return []

@st.cache_data
def load_all_structures(dataset_files: list):
    """Loads column structures for the given list of dataset files."""
    structure = {}
    if not dataset_files:
        return structure
    for fname in dataset_files:
        fpath = os.path.join('datasets', fname)
        try:
            df_peek = pd.read_csv(fpath, nrows=5)
            structure[fname] = {
                "columns": list(df_peek.columns),
                "num_cols": len(df_peek.columns)
            }
        except pd.errors.EmptyDataError:
            logger.warning(f"Dataset '{fname}' is empty. It has no columns.")
            structure[fname] = {"columns": [], "num_cols": 0}
        except Exception as e:
            logger.error(f"Error loading structure for {fname}: {e}", exc_info=True)
            structure[fname] = None
    return structure

# --- Function to Clean Code Snippet ---
def clean_code_snippet(code_input: str) -> str:
    if not isinstance(code_input, str):
        return ""
    code_str = code_input
    idx_python_block_start = code_str.find("```python")
    if idx_python_block_start != -1:
        content_start_offset = idx_python_block_start + len("```python")
        code_after_header = code_str[content_start_offset:]
        idx_end_block = code_after_header.find("```")
        extracted_code = code_after_header[:idx_end_block] if idx_end_block != -1 else code_after_header
        return extracted_code.strip()
    idx_simple_block_start = code_str.find("```")
    if idx_simple_block_start != -1:
        content_start_offset = idx_simple_block_start + len("```")
        code_after_header = code_str[content_start_offset:]
        idx_end_block = code_after_header.find("```")
        extracted_code = code_after_header[:idx_end_block] if idx_end_block != -1 else code_after_header
        return extracted_code.strip()
    return code_str.strip()


# --- Initialize Session State ---
def init_state():
    defaults = {
        'last_query': "",
        'generated_code': None,
        'execution_output': None,
        'execution_error': None,
        'is_visualization': False,
        'query_input': "",  
        'agent_selected_files': [],
        'agent_type': None,
        'agent_plan': None,
        'confirm_clear': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
# Check if conversation memory is initialized
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", input_key="query", return_messages=False,
        human_prefix="Human", ai_prefix="AI"
    )
    logger.info("Initialized new conversation memory for session.")

# --- Callback Function for Clearing State ---
def clear_state_and_memory():
    keys_to_clear = [
        'last_query','generated_code','execution_output','execution_error',
        'is_visualization','query_input','agent_selected_files','agent_type', 'agent_plan'
    ]
    for key in keys_to_clear:
        # Reset 'query_input' to empty string, others to None or False as appropriate
        if key == 'query_input':
            st.session_state[key] = ""
        elif key == 'is_visualization':
            st.session_state[key] = False
        else:
            st.session_state[key] = None

    if 'conversation_memory' in st.session_state:
        st.session_state.conversation_memory.clear()
    st.session_state.confirm_clear = False
    logger.info("Cleared conversation memory and results.")
    st.rerun()

# --- App Title and Introduction ---
logo_path = "noipa.png"  # Assumes 'logo.png' is in the same directory as the script

if os.path.exists(logo_path):
    col_logo, col_title = st.columns([1, 6]) 
    with col_logo:
        st.image(logo_path, width=150) 
    with col_title:
        st.title("NOIPA Data Assistant")
else:
    # Fallback if logo.png is not found
    st.title("NOIPA Data Assistant")

st.markdown(
    "Submit natural-language queries to receive instant, data-driven insights and visualizations."
)
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("📚 Available Datasets")
    if st.button("🔄 Refresh Dataset List", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    dataset_files = get_dataset_filenames()
    if not dataset_files:
        st.warning("No CSV datasets found in the 'datasets' folder. Please add some to proceed.")
        st.stop()
    all_structures = load_all_structures(dataset_files)
    valid_structures_info = {fname: info for fname, info in all_structures.items() if info is not None and info.get("columns") is not None}
    if not valid_structures_info:
        st.error("Could not load column structures for any dataset. Please check CSV files and logs.")
        st.stop()
    st.markdown("The system will automatically choose from these datasets:")
    for fname, info in valid_structures_info.items():
        with st.expander(f"📄 {fname} ({info['num_cols']} columns)", expanded=False):
            if not info["columns"]:
                st.info("Dataset is empty or has no columns.")
            else:
                st.markdown(f"**Columns:** `{', '.join(info['columns'])}`")
    st.markdown("---")
    st.header("🧠 Conversation")
    if st.button("🧹 Clear Conversation & Results", use_container_width=True, type="secondary"):
        st.session_state.confirm_clear = True
    if st.session_state.confirm_clear:
        st.warning("Are you sure you want to clear the conversation and all results?")
        c1, c2 = st.columns(2)
        if c1.button("✅ Yes, Clear All", use_container_width=True, type="primary"): 
            clear_state_and_memory()
        if c2.button("❌ Cancel", use_container_width=True):
            st.session_state.confirm_clear = False
            st.rerun()
    with st.expander("📜 Memory Log", expanded=False):
        if 'conversation_memory' in st.session_state:
            memory_vars = st.session_state.conversation_memory.load_memory_variables({})
            history_str = memory_vars.get("chat_history", "")
            if not history_str.strip():
                st.info("Conversation memory is empty.")
            else:
                lines = history_str.strip().split('\n')
                for line in lines:
                    escaped_line = html.escape(line)
                    if escaped_line.startswith("Human:"):
                        human_msg = escaped_line[len("Human: "):].strip()
                        st.markdown(f"<div style='text-align: right; margin-bottom: 8px; margin-left: 20px;'><span style='background-color: #DCF8C6; color: #303030; padding: 8px 12px; border-radius: 15px 15px 5px 15px; display: inline-block; max-width: 90%;'>{human_msg}</span> 🧑‍💻</div>", unsafe_allow_html=True)
                    elif escaped_line.startswith("AI:"):
                        ai_msg = escaped_line[len("AI: "):].strip()
                        st.markdown(f"<div style='text-align: left; margin-bottom: 8px; margin-right: 20px;'><span style='background-color: #ECEFF1; color: #303030; padding: 8px 12px; border-radius: 15px 15px 15px 5px; display: inline-block; max-width: 90%;'>{ai_msg}</span> 🤖</div>", unsafe_allow_html=True)
                    else:
                        st.text(line)
        else:
            st.info("Conversation memory not initialized.")

# --- Main Area ---

# --- Callback for example query buttons ---
def set_query_from_example(example_query_text):
    st.session_state.query_input = example_query_text

st.subheader("💬 Your Query")
current_query_in_box = st.text_input(
    "Enter your request:",
    key="query_input", # 
    placeholder="E.g., 'What is the average salary by department, merging employee and salary data?'",
    label_visibility="collapsed"
)

st.markdown("<small>💡 Example Queries:</small>", unsafe_allow_html=True)
example_queries = [
    "Plot a barchart  of the top 10 regions for number of occurrences.",
    "List all the distinct age groups.",
    "Plot a Piechart with the distribution of the authentication_method "
]
cols = st.columns(len(example_queries))
for i, ex_query in enumerate(example_queries):
    cols[i].button(
        ex_query,
        use_container_width=True,
        key=f"ex_query_{i}",
        on_click=set_query_from_example, 
        args=(ex_query,)
    )

run_button = st.button("🚀 Process Query", use_container_width=True, type="primary")

# --- Execution Logic ---
if run_button and current_query_in_box: 
    st.session_state.last_query = current_query_in_box
    st.session_state.generated_code = None
    st.session_state.execution_output = None
    st.session_state.execution_error = None
    st.session_state.is_visualization = False
    st.session_state.agent_selected_files = []
    st.session_state.agent_type = None
    st.session_state.agent_plan = None

    with st.status("Agent at work... 🕵️", expanded=True) as status:
        try:
            status.update(label="Understanding your query... 🤔", state="running")
            
            orchestrator_response = orchestrate(
                query=st.session_state.last_query,
                all_dataset_structures={fname: info["columns"] for fname, info in valid_structures_info.items()},
                memory=st.session_state.conversation_memory
            )

            if "error" in orchestrator_response:
                st.session_state.execution_error = orchestrator_response["error"]
                status.update(label=f"⚠️ Orchestrator Error: {st.session_state.execution_error}", state="error")
            else:
                st.session_state.agent_selected_files = orchestrator_response.get("selected_files", [])
                st.session_state.agent_type = orchestrator_response.get("agent_type")
                st.session_state.agent_plan = orchestrator_response.get("plan")
                raw_code = orchestrator_response.get("code")
                st.session_state.generated_code = raw_code
                
                if not raw_code:
                    st.session_state.execution_error = "Agent did not return any code to execute."
                    status.update(label="⚠️ Agent returned no code", state="error")
                else:
                    code_to_execute = clean_code_snippet(raw_code)
                    st.session_state.is_visualization = (st.session_state.agent_type == "visualization")

                    status.update(label="Executing generated code... ⚙️", state="running")
                    if st.session_state.is_visualization:
                        plt.close('all')
                        exec_globals = dict(python_repl_globals)
                        try:
                            exec(code_to_execute, exec_globals)
                            if plt.get_fignums():
                                fig = plt.gcf()
                                if fig.get_axes():
                                    st.session_state.execution_output = fig
                                else:
                                    st.session_state.execution_output = "Visualization code ran, but the plot is empty."
                                    st.session_state.is_visualization = False
                            else:
                                st.session_state.execution_output = "Visualization code ran, but no plot was generated."
                                st.session_state.is_visualization = False
                        except Exception as exec_e:
                            logger.error(f"Error during exec for visualization: {exec_e}", exc_info=True)
                            st.session_state.execution_error = f"Error executing visualization code: {exec_e}\n{traceback.format_exc()}"
                            st.session_state.is_visualization = False
                    else:
                        dataframe_var_name = "dataframe_output"
                        if dataframe_var_name in python_repl.locals:
                            del python_repl.locals[dataframe_var_name]

                        repl_stdout_output = python_repl.run(tool_input=code_to_execute, intermediate_steps=[])

                        if dataframe_var_name in python_repl.locals and \
                           isinstance(python_repl.locals[dataframe_var_name], pd.DataFrame):
                            st.session_state.execution_output = python_repl.locals[dataframe_var_name]
                            if repl_stdout_output and repl_stdout_output.strip():
                                logger.info(f"REPL stdout (DataFrame also produced and prioritized):\n{repl_stdout_output.strip()}")
                        else:
                            st.session_state.execution_output = repl_stdout_output.strip() if repl_stdout_output else "Code executed, but no textual output or DataFrame named 'dataframe_output' was produced."
                
                if st.session_state.execution_error:
                     status.update(label="❌ Error during execution!", state="error")
                else:
                    status.update(label="✅ Processing complete!", state="complete")

        except Exception as e:
            st.session_state.execution_error = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Unexpected error in main execution block: {e}", exc_info=True)
            status.update(label="❌ Unexpected error!", state="error")

# --- Display Results ---
if st.session_state.last_query:
    st.markdown("---")
    st.subheader("📊 Agent Results")

    if st.session_state.agent_selected_files:
        files_str = ", ".join(f"`{f}`" for f in st.session_state.agent_selected_files)
        agent_action_type = st.session_state.agent_type or "task"
        st.markdown(f"ℹ️ **Agent Action:** Performed a `{agent_action_type}` operation using: {files_str}")

    if st.session_state.agent_plan:
        with st.expander("🤖 Agent's Plan", expanded=False):
            st.markdown(st.session_state.agent_plan)

    tab_titles = []
    if st.session_state.is_visualization and isinstance(st.session_state.execution_output, plt.Figure):
        tab_titles.append("📊 Visualization")
    if isinstance(st.session_state.execution_output, pd.DataFrame):
        tab_titles.append("📄 DataFrame")
    if isinstance(st.session_state.execution_output, str) and not st.session_state.execution_error :
        tab_titles.append("📝 Text Output")
    is_other_output = (
        st.session_state.execution_output is not None and
        not st.session_state.is_visualization and
        not isinstance(st.session_state.execution_output, (pd.DataFrame, str)) and
        not st.session_state.execution_error
    )
    if is_other_output:
        tab_titles.append("💡 Other Output")
    if st.session_state.generated_code:
        tab_titles.append("🐍 Generated Code")
    if st.session_state.execution_error:
        tab_titles.append("⚠️ Error Details")

    if not tab_titles and not st.session_state.execution_error:
        st.info("Processing complete. No specific output or code to display for this query.")
    elif tab_titles:
        tabs = st.tabs(tab_titles)
        current_tab_idx = 0

        if "📊 Visualization" in tab_titles:
            with tabs[current_tab_idx]:
                st.subheader(f"Visualization Output ({st.session_state.agent_type or 'Plot'})")
                st.pyplot(st.session_state.execution_output)
                try:
                    plot_bytes = BytesIO()
                    st.session_state.execution_output.savefig(plot_bytes, format='png', bbox_inches='tight')
                    plot_bytes.seek(0)
                    st.download_button(
                        label="📥 Download Plot (PNG)",
                        data=plot_bytes,
                        file_name=f"plot_{st.session_state.last_query[:20].replace(' ','_')}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.warning(f"Could not generate download for plot: {e}")
            current_tab_idx += 1
        if "📄 DataFrame" in tab_titles:
            with tabs[current_tab_idx]:
                st.subheader(f"DataFrame Output ({st.session_state.agent_type or 'Table'})")
                df_len = len(st.session_state.execution_output) if isinstance(st.session_state.execution_output, pd.DataFrame) else 0
                df_height = min(600, (df_len + 1) * 35 + 3) if df_len > 0 else 100
                st.dataframe(st.session_state.execution_output, use_container_width=True, height=df_height)
                try:
                    csv = st.session_state.execution_output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Data (CSV)",
                        data=csv,
                        file_name=f"data_{st.session_state.last_query[:20].replace(' ','_')}.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.warning(f"Could not generate download for CSV: {e}")
            current_tab_idx += 1

        if "📝 Text Output" in tab_titles:
            with tabs[current_tab_idx]:
                st.subheader(f"Text Output ({st.session_state.agent_type or 'Info'})")
                st.text_area("Result", st.session_state.execution_output, height=300, key="text_output_main_area_tab")
            current_tab_idx += 1
        
        if "💡 Other Output" in tab_titles:
            with tabs[current_tab_idx]:
                st.subheader(f"Other Output ({st.session_state.agent_type or 'Result'})")
                st.write("Execution produced non-standard output:")
                st.write(st.session_state.execution_output)
            current_tab_idx += 1

        if "🐍 Generated Code" in tab_titles:
            with tabs[current_tab_idx]:
                st.subheader("Generated Python Code")
                st.code(st.session_state.generated_code, language='python')
                st.download_button(
                    label="💾 Download Code (.py)",
                    data=str(st.session_state.generated_code),
                    file_name="generated_code.py",
                    mime="text/x-python",
                )
            current_tab_idx += 1
        
        if "⚠️ Error Details" in tab_titles:
            with tabs[current_tab_idx]:
                st.text_area("Error Information", st.session_state.execution_error, height=300, key="error_details_area_tab")
                st.download_button(
                    label="📋 Download Error Details",
                    data=str(st.session_state.execution_error),
                    file_name="error_details.txt",
                    mime="text/plain",
                )
            current_tab_idx += 1
    elif st.session_state.execution_error:
        st.error("An error occurred:")
        st.text_area("Error Details", st.session_state.execution_error, height=200, key="error_details_area_standalone")

st.markdown("---")
st.caption("Powered by Langchain, Ollama, and Streamlit. System autonomously selects datasets, plans, and executes. Built by SQPR Consulting.")

